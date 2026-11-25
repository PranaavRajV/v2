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

# ── Serve static build with Nginx on Railway ────────────────────────────────
FROM nginx:alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html

RUN printf 'server {\n\
  __LISTEN__\n\
  root /usr/share/nginx/html;\n\
  index index.html;\n\
  location / {\n\
    try_files $uri $uri/ /index.html;\n\
  }\n\
}\n' > /etc/nginx/conf.d/default.conf.template

EXPOSE 80

CMD ["sh", "-c", "if [ -n \"$PORT\" ] && [ \"$PORT\" != \"80\" ]; then LISTEN=\"listen 80; listen [::]:80; listen ${PORT}; listen [::]:${PORT};\"; else LISTEN=\"listen 80; listen [::]:80;\"; fi; sed \"s|__LISTEN__|${LISTEN}|g\" /etc/nginx/conf.d/default.conf.template > /etc/nginx/conf.d/default.conf && nginx -g 'daemon off;'"]
