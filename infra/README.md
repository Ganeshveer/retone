# Infra setup — Cloudflare R2

## 1. Create the bucket

Cloudflare dashboard → **R2** → *Create bucket* → name it `retone` (matches `R2_BUCKET`).

## 2. Create an R2 API token

R2 → *Manage R2 API Tokens* → *Create API Token* → **Object Read & Write** scoped to the
`retone` bucket. Copy into `backend/.env`:

```
R2_ACCOUNT_ID=<your account id>
R2_ACCESS_KEY_ID=<token access key id>
R2_SECRET_ACCESS_KEY=<token secret>
R2_BUCKET=retone
```

The S3 endpoint is `https://<R2_ACCOUNT_ID>.r2.cloudflarestorage.com` (the backend derives it).

## 3. CORS (so the browser can PUT/GET directly)

Bucket → *Settings* → *CORS policy* → paste (replace the origin for production):

```json
[
  {
    "AllowedOrigins": ["http://localhost:5173"],
    "AllowedMethods": ["PUT", "GET", "HEAD"],
    "AllowedHeaders": ["content-type"],
    "ExposeHeaders": ["ETag"],
    "MaxAgeSeconds": 3600
  }
]
```

Notes:
- The browser uploads with `Content-Type` matching what the backend signed — keep
  `AllowedHeaders: ["content-type"]` (not `*`, which can 403 some presigned PUTs).
- The RunPod worker talks to R2 with its own keys (server-side), so it isn't subject to CORS.

## 4. Flip the backend to live mode

In `backend/.env`: set `MOCK_MODE=false` and fill in the R2 + RunPod values, then restart
`uvicorn`. `GET /health` should report `"r2_configured": true, "runpod_configured": true`.

## 5. (Optional) public download domain

If you'd rather serve stems from a public URL than presigned GETs, attach a custom domain
to the bucket and serve `stems/*` from it. For now the backend hands out short-lived
presigned GET URLs, which needs no public domain.
