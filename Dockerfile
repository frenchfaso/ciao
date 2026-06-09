FROM node:22-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
COPY scripts/require-dev-container.mjs ./scripts/require-dev-container.mjs
RUN npm ci

COPY . .
RUN npm run build

FROM golang:1.23-alpine AS server-build

WORKDIR /src
COPY go.mod ./
COPY deploy/server ./deploy/server
RUN CGO_ENABLED=0 GOOS=linux go build -trimpath -ldflags="-s -w" -o /out/ciao-server ./deploy/server

FROM alpine:3.20 AS runtime

RUN addgroup -S ciao && adduser -S -G ciao ciao
WORKDIR /app

COPY --from=build /app/dist /app/public
COPY --from=server-build /out/ciao-server /app/ciao-server

USER ciao
EXPOSE 8080
ENV CIAO_ADDR=:8080
ENV CIAO_STATIC_DIR=/app/public

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD wget -qO- http://127.0.0.1:8080/healthz >/dev/null || exit 1

CMD ["/app/ciao-server"]
