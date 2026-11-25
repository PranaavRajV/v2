# ── DermaCLIP Frontend (React + Vite) ──────────────────────────────────────
FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .

# Railway passes service Variables as build ARGs automatically
ARG VITE_API_BASE=http://localhost:8081
ENV VITE_API_BASE=$VITE_API_BASE

RUN npm run build

# ── Serve static build with Nginx on Railway $PORT ──────────────────────────
FROM nginx:alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html

# SPA fallback + dynamic $PORT binding for Railway
RUN printf 'server {\n\
  listen __PORT__;\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf.template

ENV PORT=80
EXPOSE 80

CMD ["sh", "-c", "sed \"s/__PORT__/${PORT:-80}/g\" /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
