package main

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"mime"
	"net"
	"net/http"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"time"
)

const (
	webSocketGUID       = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
	maxMessage          = 64 * 1024
	roomTTL             = 10 * time.Minute
	sendQueueDepth      = 64
	sendQueueTimeout    = 2 * time.Second
	webSocketReadIdle   = 75 * time.Second
	webSocketWriteLimit = 10 * time.Second
	webSocketPingEvery  = 25 * time.Second
)

var roomPattern = regexp.MustCompile(`^[a-zA-Z0-9_-]{6,48}$`)

func main() {
	_ = mime.AddExtensionType(".mjs", "text/javascript; charset=utf-8")
	_ = mime.AddExtensionType(".onnx", "application/octet-stream")
	_ = mime.AddExtensionType(".data", "application/octet-stream")
	_ = mime.AddExtensionType(".wasm", "application/wasm")

	staticDir := env("CIAO_STATIC_DIR", "/app/public")
	addr := env("CIAO_ADDR", ":8080")
	hub := newHub()

	go hub.cleanupLoop()

	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", healthHandler)
	mux.HandleFunc("/ws", hub.handleWebSocket)
	mux.HandleFunc("/echo", echoHandler)
	mux.HandleFunc("/", staticHandler(staticDir))

	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	log.Printf("ciao server listening on %s", addr)
	log.Fatal(server.ListenAndServe())
}

func env(key string, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}

	return fallback
}

func healthHandler(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	_, _ = w.Write([]byte("ok\n"))
}

func echoHandler(w http.ResponseWriter, r *http.Request) {
	conn, err := acceptWebSocket(w, r)
	if err != nil {
		log.Printf("echo websocket upgrade failed: %v", err)
		return
	}
	defer conn.close()

	for {
		opcode, payload, err := conn.readFrame()
		if err != nil {
			return
		}

		switch opcode {
		case 0x1:
			if err := conn.writeText(payload); err != nil {
				return
			}
		case 0x2:
			if err := conn.writeBinary(payload); err != nil {
				return
			}
		case 0x8:
			return
		case 0x9:
			_ = conn.writeFrame(0xA, payload)
		case 0xA:
			continue
		default:
			return
		}
	}
}

func staticHandler(staticDir string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		addSecurityHeaders(w)

		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		cleanPath := path.Clean("/" + strings.TrimPrefix(r.URL.Path, "/"))
		if cleanPath == "/bench" || strings.HasPrefix(cleanPath, "/bench/") {
			http.NotFound(w, r)
			return
		}

		if cleanPath == "/" {
			serveFile(w, r, filepath.Join(staticDir, "index.html"), "no-store")
			return
		}

		filePath := filepath.Join(staticDir, filepath.FromSlash(strings.TrimPrefix(cleanPath, "/")))
		if info, err := os.Stat(filePath); err == nil && !info.IsDir() {
			serveFile(w, r, filePath, cachePolicy(cleanPath))
			return
		}

		if path.Ext(cleanPath) != "" {
			http.NotFound(w, r)
			return
		}

		serveFile(w, r, filepath.Join(staticDir, "index.html"), "no-store")
	}
}

func serveFile(w http.ResponseWriter, r *http.Request, filePath string, cacheControl string) {
	if contentType := mime.TypeByExtension(filepath.Ext(filePath)); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}

	w.Header().Set("Cache-Control", cacheControl)
	http.ServeFile(w, r, filePath)
}

func cachePolicy(urlPath string) string {
	if urlPath == "/index.html" || urlPath == "/sw.js" || strings.HasPrefix(urlPath, "/workbox-") {
		return "no-store"
	}

	if urlPath == "/manifest.webmanifest" {
		return "no-cache"
	}

	if strings.HasPrefix(urlPath, "/ort/") ||
		strings.HasPrefix(urlPath, "/models/mimi/streaming-8cb-fp16/") {
		return "public, max-age=31536000, immutable"
	}

	if strings.HasPrefix(urlPath, "/assets/") {
		return "public, max-age=31536000, immutable"
	}

	switch strings.ToLower(path.Ext(urlPath)) {
	case ".svg", ".png", ".ico", ".webp", ".wasm", ".onnx", ".data", ".bin", ".mjs", ".js", ".css", ".txt":
		return "public, max-age=86400"
	default:
		return "no-store"
	}
}

func addSecurityHeaders(w http.ResponseWriter) {
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("Referrer-Policy", "no-referrer")
	w.Header().Set("Permissions-Policy", "camera=(), geolocation=(), microphone=(self)")
	w.Header().Set("Cross-Origin-Opener-Policy", "same-origin")
	w.Header().Set("Cross-Origin-Embedder-Policy", "require-corp")
}

type hub struct {
	mu    sync.Mutex
	rooms map[string]*room
}

type room struct {
	clients    map[string]*client
	lastActive time.Time
}

type client struct {
	hub       *hub
	conn      *webSocketConn
	room      string
	role      string
	send      chan []byte
	done      chan struct{}
	closeOnce sync.Once
}

func newHub() *hub {
	return &hub{rooms: make(map[string]*room)}
}

func (h *hub) handleWebSocket(w http.ResponseWriter, r *http.Request) {
	roomID := r.URL.Query().Get("room")
	role := r.URL.Query().Get("role")

	if !roomPattern.MatchString(roomID) || (role != "host" && role != "guest") {
		http.Error(w, "invalid signaling room", http.StatusBadRequest)
		return
	}

	conn, err := acceptWebSocket(w, r)
	if err != nil {
		log.Printf("websocket upgrade failed: %v", err)
		return
	}

	c := &client{
		hub:  h,
		conn: conn,
		room: roomID,
		role: role,
		send: make(chan []byte, sendQueueDepth),
		done: make(chan struct{}),
	}

	if err := h.register(c); err != nil {
		_ = conn.writeJSON(map[string]string{"type": "error", "message": err.Error()})
		_ = conn.close()
		return
	}

	go c.writeLoop()
	c.readLoop()
}

func (h *hub) register(c *client) error {
	var peers []*client

	h.mu.Lock()

	r := h.rooms[c.room]
	if r == nil {
		r = &room{
			clients:    make(map[string]*client),
			lastActive: time.Now(),
		}
		h.rooms[c.room] = r
	}

	if r.clients[c.role] != nil {
		h.mu.Unlock()
		return errors.New("ruolo gia' presente")
	}

	r.clients[c.role] = c
	r.lastActive = time.Now()

	for _, peer := range r.clients {
		if peer != c {
			peers = append(peers, peer)
		}
	}
	h.mu.Unlock()

	if err := queueJSON(c, map[string]string{"type": "joined", "role": c.role, "room": c.room}); err != nil {
		h.unregister(c)
		return err
	}

	for _, peer := range peers {
		if err := queueJSON(peer, map[string]string{"type": "peer-joined", "role": c.role}); err != nil {
			log.Printf("signaling notify failed for room %s role %s: %v", peer.room, peer.role, err)
			go h.unregister(peer)
		}
		if err := queueJSON(c, map[string]string{"type": "peer-joined", "role": peer.role}); err != nil {
			h.unregister(c)
			return err
		}
	}

	return nil
}

func (h *hub) unregister(c *client) {
	var peers []*client
	removed := false

	h.mu.Lock()
	if r := h.rooms[c.room]; r != nil {
		if r.clients[c.role] == c {
			delete(r.clients, c.role)
			r.lastActive = time.Now()
			removed = true

			for _, peer := range r.clients {
				peers = append(peers, peer)
			}

			if len(r.clients) == 0 {
				delete(h.rooms, c.room)
			}
		}
	}
	h.mu.Unlock()

	c.close()

	if !removed {
		return
	}

	for _, peer := range peers {
		if err := queueJSON(peer, map[string]string{"type": "peer-left", "role": c.role}); err != nil {
			log.Printf("signaling leave notify failed for room %s role %s: %v", peer.room, peer.role, err)
			go h.unregister(peer)
		}
	}
}

func (h *hub) forward(sender *client, payload []byte) {
	var peers []*client

	h.mu.Lock()

	r := h.rooms[sender.room]
	if r == nil {
		h.mu.Unlock()
		return
	}

	r.lastActive = time.Now()
	for _, peer := range r.clients {
		if peer != sender {
			peers = append(peers, peer)
		}
	}
	h.mu.Unlock()

	for _, peer := range peers {
		if err := queueBytes(peer, payload); err != nil {
			log.Printf("signaling forward failed for room %s role %s: %v", peer.room, peer.role, err)
			go h.unregister(peer)
		}
	}
}

func (h *hub) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()

	for range ticker.C {
		cutoff := time.Now().Add(-roomTTL)
		h.mu.Lock()
		for id, room := range h.rooms {
			if len(room.clients) == 0 && room.lastActive.Before(cutoff) {
				delete(h.rooms, id)
			}
		}
		h.mu.Unlock()
	}
}

func (c *client) readLoop() {
	defer c.hub.unregister(c)

	for {
		payload, err := c.conn.readText()
		if err != nil {
			return
		}

		c.hub.forward(c, payload)
	}
}

func (c *client) writeLoop() {
	ticker := time.NewTicker(webSocketPingEvery)
	defer ticker.Stop()

	for {
		select {
		case payload := <-c.send:
			if err := c.conn.writeText(payload); err != nil {
				c.hub.unregister(c)
				return
			}
		case <-ticker.C:
			if err := c.conn.writeFrame(0x9, nil); err != nil {
				c.hub.unregister(c)
				return
			}
		case <-c.done:
			return
		}
	}
}

func (c *client) close() {
	c.closeOnce.Do(func() {
		close(c.done)
		_ = c.conn.close()
	})
}

func queueJSON(c *client, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}

	return queueBytes(c, payload)
}

func queueBytes(c *client, payload []byte) error {
	select {
	case <-c.done:
		return errors.New("client closed")
	default:
	}

	timer := time.NewTimer(sendQueueTimeout)
	defer timer.Stop()

	select {
	case c.send <- payload:
		return nil
	case <-c.done:
		return errors.New("client closed")
	case <-timer.C:
		return errors.New("client send queue timeout")
	}
}

type webSocketConn struct {
	conn         net.Conn
	reader       *bufio.Reader
	writer       *bufio.Writer
	readTimeout  time.Duration
	writeTimeout time.Duration
	mu           sync.Mutex
}

func acceptWebSocket(w http.ResponseWriter, r *http.Request) (*webSocketConn, error) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return nil, errors.New("invalid websocket method")
	}

	if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		http.Error(w, "upgrade required", http.StatusUpgradeRequired)
		return nil, errors.New("missing websocket upgrade")
	}

	if !headerContainsToken(r.Header.Get("Connection"), "upgrade") {
		http.Error(w, "upgrade required", http.StatusUpgradeRequired)
		return nil, errors.New("missing websocket connection upgrade")
	}

	if r.Header.Get("Sec-WebSocket-Version") != "13" {
		http.Error(w, "unsupported websocket version", http.StatusBadRequest)
		return nil, errors.New("unsupported websocket version")
	}

	key := r.Header.Get("Sec-WebSocket-Key")
	decodedKey, err := base64.StdEncoding.DecodeString(key)
	if key == "" || err != nil || len(decodedKey) != 16 {
		http.Error(w, "missing websocket key", http.StatusBadRequest)
		return nil, errors.New("invalid websocket key")
	}

	hijacker, ok := w.(http.Hijacker)
	if !ok {
		http.Error(w, "hijacking unsupported", http.StatusInternalServerError)
		return nil, errors.New("hijacking unsupported")
	}

	conn, rw, err := hijacker.Hijack()
	if err != nil {
		return nil, err
	}

	accept := webSocketAccept(key)
	_, err = fmt.Fprintf(
		rw,
		"HTTP/1.1 101 Switching Protocols\r\n"+
			"Upgrade: websocket\r\n"+
			"Connection: Upgrade\r\n"+
			"Sec-WebSocket-Accept: %s\r\n\r\n",
		accept,
	)
	if err != nil {
		_ = conn.Close()
		return nil, err
	}

	if err := rw.Flush(); err != nil {
		_ = conn.Close()
		return nil, err
	}

	return &webSocketConn{
		conn:         conn,
		reader:       rw.Reader,
		writer:       rw.Writer,
		readTimeout:  webSocketReadIdle,
		writeTimeout: webSocketWriteLimit,
	}, nil
}

func headerContainsToken(value string, token string) bool {
	for _, part := range strings.Split(value, ",") {
		if strings.EqualFold(strings.TrimSpace(part), token) {
			return true
		}
	}

	return false
}

func webSocketAccept(key string) string {
	hash := sha1.Sum([]byte(key + webSocketGUID))
	return base64.StdEncoding.EncodeToString(hash[:])
}

func (c *webSocketConn) readText() ([]byte, error) {
	for {
		opcode, payload, err := c.readFrame()
		if err != nil {
			return nil, err
		}

		switch opcode {
		case 0x1:
			return payload, nil
		case 0x8:
			return nil, io.EOF
		case 0x9:
			_ = c.writeFrame(0xA, payload)
		case 0xA:
			continue
		}
	}
}

func (c *webSocketConn) readFrame() (byte, []byte, error) {
	if c.readTimeout > 0 {
		_ = c.conn.SetReadDeadline(time.Now().Add(c.readTimeout))
	}

	header := make([]byte, 2)
	if _, err := io.ReadFull(c.reader, header); err != nil {
		return 0, nil, err
	}

	fin := header[0]&0x80 != 0
	opcode := header[0] & 0x0F
	masked := header[1]&0x80 != 0
	length := uint64(header[1] & 0x7F)

	if !fin {
		return 0, nil, errors.New("fragmented websocket frames are not supported")
	}

	if !masked {
		return 0, nil, errors.New("unmasked websocket client frame")
	}

	if length == 126 {
		extended := make([]byte, 2)
		if _, err := io.ReadFull(c.reader, extended); err != nil {
			return 0, nil, err
		}
		length = uint64(binary.BigEndian.Uint16(extended))
	} else if length == 127 {
		extended := make([]byte, 8)
		if _, err := io.ReadFull(c.reader, extended); err != nil {
			return 0, nil, err
		}
		length = binary.BigEndian.Uint64(extended)
	}

	if length > maxMessage {
		return 0, nil, errors.New("websocket message too large")
	}

	var maskKey [4]byte
	if masked {
		if _, err := io.ReadFull(c.reader, maskKey[:]); err != nil {
			return 0, nil, err
		}
	}

	payload := make([]byte, length)
	if _, err := io.ReadFull(c.reader, payload); err != nil {
		return 0, nil, err
	}

	if masked {
		for i := range payload {
			payload[i] ^= maskKey[i%4]
		}
	}

	return opcode, payload, nil
}

func (c *webSocketConn) writeJSON(value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}

	return c.writeText(payload)
}

func (c *webSocketConn) writeText(payload []byte) error {
	return c.writeFrame(0x1, payload)
}

func (c *webSocketConn) writeBinary(payload []byte) error {
	return c.writeFrame(0x2, payload)
}

func (c *webSocketConn) writeFrame(opcode byte, payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.writeTimeout > 0 {
		_ = c.conn.SetWriteDeadline(time.Now().Add(c.writeTimeout))
	}

	header := []byte{0x80 | opcode}
	length := len(payload)

	if length < 126 {
		header = append(header, byte(length))
	} else if length <= 0xFFFF {
		header = append(header, 126, byte(length>>8), byte(length))
	} else {
		header = append(header, 127)
		extended := make([]byte, 8)
		binary.BigEndian.PutUint64(extended, uint64(length))
		header = append(header, extended...)
	}

	if _, err := c.writer.Write(header); err != nil {
		return err
	}

	if _, err := c.writer.Write(payload); err != nil {
		return err
	}

	return c.writer.Flush()
}

func (c *webSocketConn) close() error {
	return c.conn.Close()
}
