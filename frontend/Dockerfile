# ---- build stage: install deps and produce the static bundle ----
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN if [ -f package-lock.json ]; then npm ci; else npm install; fi
COPY . .
RUN npm run build

# ---- runtime stage: static files served by nginx on port 3000 ----
FROM nginx:1.27-alpine AS runtime
ENV API_BASE_URL=http://localhost:8000
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY docker-entrypoint.sh /docker-entrypoint.d/40-app-config.sh
RUN chmod +x /docker-entrypoint.d/40-app-config.sh
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 3000
