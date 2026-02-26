# SSL/TLS Deployment Configuration Guide

## Problem

At startup:
```
Running without SSL
```

This indicates:
- Plain HTTP mode (no encryption)
- No TLS certificates
- No HTTPS listener

## Security Assessment

### Your Current Setup

From logs, server is bound to:
```
Default: 127.0.0.1:8089  (local-only)
```

**Risk Level**: ✅ **LOW** (Informational only)

**Why**:
- 127.0.0.1 is localhost (accessible only from local machine)
- No external exposure
- No network transmission risk
- No MITM attack surface

### If You Change Binding

**High Risk Scenarios**:

```
./llama-server --host 0.0.0.0      ✗ Exposes to all interfaces
./llama-server --host 192.168.1.0  ✗ Exposes to LAN
./llama-server --listen 0.0.0.0    ✗ Public internet exposure
```

In these cases:
- ❌ Prompts sent unencrypted
- ❌ API responses visible on network
- ❌ Vulnerable to man-in-the-middle (MITM) attacks
- ❌ Model outputs observable by network sniffing

## Why SSL Is Disabled by Default

```
llama-server design:
├─ Designed for local/containerized deployments
├─ TLS termination delegated to reverse proxy
├─ Avoids certificate management in app
└─ Simpler architecture (app handles logic, proxy handles encryption)

Common assumption:
"I only run this locally, why add complexity?"
```

## Proper Production Architecture

### ❌ Wrong (Direct HTTPS in app)

```
Client → HTTPS → llama-server [TLS handling, certificate management]
```

Problems:
- Certificate renewal requires app restart
- Duplicate encryption logic
- Complex configuration
- Hard to scale

### ✅ Correct (Reverse Proxy TLS Termination)

```
Client → HTTPS (TLS encrypted) → Reverse Proxy (Nginx/Caddy)
                                     ↓
                              HTTP (local only)
                                     ↓
                              llama-server:8089
```

Benefits:
- ✓ Mature TLS implementation (Nginx/Caddy)
- ✓ Automatic certificate renewal (Let's Encrypt)
- ✓ Rate limiting
- ✓ Access control
- ✓ Logging and monitoring
- ✓ Easy scaling (load balancing)
- ✓ llama-server stays simple

## Setup Guides

### Scenario 1: Local Development (Current)

**Current setup is fine**:
```bash
./llama-server -m model.gguf -ngl 999 --host 127.0.0.1 --port 8089
```

**Status**: ✅ Secure (local-only)

No changes needed.

### Scenario 2: LAN Access (Small Network)

**Option A: Still use reverse proxy (recommended)**

Install Caddy (simple, automatic HTTPS):
```bash
# Install caddy
curl https://getcaddy.com | bash

# Create Caddyfile
cat > Caddyfile << 'EOF'
your-local-domain.local {
    reverse_proxy 127.0.0.1:8089
}
EOF

# Start caddy
caddy run
```

**Access**: `https://your-local-domain.local`

**Option B: Self-signed certificates (not recommended)**

```bash
# Generate self-signed cert (if you must)
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365

# But: llama-server doesn't support TLS
# You STILL need a reverse proxy
```

### Scenario 3: Public Internet (Production)

**MUST use reverse proxy with proper certificates**

#### Setup with Nginx

```bash
# 1. Install nginx
sudo apt-get install nginx

# 2. Install certbot (Let's Encrypt)
sudo apt-get install certbot python3-certbot-nginx

# 3. Create Nginx config
sudo cat > /etc/nginx/sites-available/llama-api << 'EOF'
server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8089;

        # Forward headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts for long requests
        proxy_connect_timeout 600s;
        proxy_send_timeout 600s;
        proxy_read_timeout 600s;
    }
}
EOF

# 4. Enable site
sudo ln -s /etc/nginx/sites-available/llama-api /etc/nginx/sites-enabled/

# 5. Test config
sudo nginx -t

# 6. Start nginx
sudo systemctl start nginx

# 7. Obtain Let's Encrypt certificate
sudo certbot --nginx -d api.yourdomain.com
```

**Result**:
```
Client → HTTPS (encrypted) → Nginx:443
                                  ↓
                          HTTP (internal only)
                                  ↓
                          llama-server:8089 (127.0.0.1)
```

#### Setup with Caddy (Simpler)

```bash
# 1. Install caddy
sudo apt-get install caddy

# 2. Create Caddyfile
sudo cat > /etc/caddy/Caddyfile << 'EOF'
api.yourdomain.com {
    reverse_proxy 127.0.0.1:8089 {
        # Forward headers
        header_down Content-Type "application/json"
    }
}
EOF

# 3. Start caddy
sudo systemctl start caddy

# Automatic:
# - HTTPS certificate obtained from Let's Encrypt
# - Certificate renewed automatically
# - Listening on ports 80/443
```

**Result**: Same as Nginx, but automated

### Scenario 4: Docker Container

**Inside container (development)**:
```bash
./llama-server -m model.gguf --host 0.0.0.0 --port 8089
# Exposing internally within container only
```

**Docker network**:
```dockerfile
FROM ubuntu:22.04
# ... install llama ...
EXPOSE 8089
CMD ["./llama-server", "-m", "model.gguf", "--host", "0.0.0.0"]
```

**Run with reverse proxy host**:
```bash
docker run -p 127.0.0.1:8089:8089 llama-server:latest

# Then run Nginx/Caddy on host to handle HTTPS
```

## API Key Security

If using API keys with llama-server:

### ❌ Without HTTPS

```
Client sends: GET /api/generate?key=sk-12345...
Transmitted: CLEARTEXT over HTTP
Observer can: Read key from network traffic
```

### ✅ With HTTPS (via reverse proxy)

```
Client sends: GET https://api.yourdomain.com/api/generate?key=sk-12345...
Transmitted: ENCRYPTED (TLS)
Observer cannot: Decrypt traffic
```

**Recommendation**: Always use HTTPS with reverse proxy if API keys are involved.

## Current Deployment Status

### Your Setup

```
Server:      127.0.0.1:8089
Binding:     Local-only (not exposed)
SSL Status:  Disabled (not needed for local)
Risk:        Low
Action:      None required
```

**Status**: ✅ Secure for local development

### If You Scale to Production

```
Before:  ./llama-server -m model.gguf
Risk:    Unencrypted, exposed

After:   Nginx/Caddy reverse proxy
         → HTTPS termination
         → Automatic certificates
         → Secure API access
Risk:    Encrypted, protected
```

## Verification Checklist

### Current (Local-only)

```bash
# Check binding
netstat -tulpn | grep llama
# Should show: 127.0.0.1:8089 (not 0.0.0.0:8089)

# Test local access
curl http://127.0.0.1:8089/health
# Works ✓
```

### Production (With Reverse Proxy)

```bash
# Check reverse proxy
sudo systemctl status nginx
# or
sudo systemctl status caddy

# Test HTTPS access
curl https://api.yourdomain.com/health
# Works with TLS ✓

# Verify certificate
curl -I https://api.yourdomain.com
# Headers should show: "Strict-Transport-Security"
```

## Summary

| Scenario | Binding | SSL | Setup | Risk |
|----------|---------|-----|-------|------|
| Local dev | 127.0.0.1 | None | Direct | ✅ None |
| LAN access | 192.168.x.x | Nginx/Caddy | Reverse proxy | ⚠️ Medium |
| Public API | 0.0.0.0 | Nginx/Caddy + Let's Encrypt | Reverse proxy | ✅ Low |

## Recommendations

### Development (You, now)
```bash
./llama-server -m model.gguf -ngl 999 --host 127.0.0.1
```
**Action**: None required. Secure for local development.

### Testing (Small LAN)
```bash
# Run with Caddy reverse proxy
caddy run  # Handles TLS automatically
```
**Action**: Add Caddy for HTTPS if exposing to LAN.

### Production (Public Internet)
```bash
# Use Nginx or Caddy with Let's Encrypt
./llama-server -m model.gguf --host 127.0.0.1  # Internal only
# Nginx/Caddy handles external HTTPS
```
**Action**: MUST use reverse proxy with proper TLS before exposing.

## Conclusion

**SSL disabled is not a vulnerability if**:
- ✅ Bound to 127.0.0.1 (local-only)
- ✅ Not exposed to network

**SSL disabled IS a vulnerability if**:
- ❌ Bound to 0.0.0.0
- ❌ Accessible from internet
- ❌ Without reverse proxy

**Current Status**: ✅ **Secure** (local development, no action needed)

**Future**: When scaling to production, add Nginx/Caddy with Let's Encrypt certificates.
