<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DeepSeek Opportunity Scanner</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='styles.css') }}">
</head>
<body>
  <main class="container">
    <section class="hero card">
      <h1>DeepSeek Opportunity Scanner</h1>
      <p>Crawl a website, skip already analyzed content, analyze fresh pages with DeepSeek, then POST the results to another website.</p>
    </section>

    <section class="card">
      <form action="/analyze" method="post" class="form-grid">
        <div class="field full">
          <label for="start_url">Website URL to crawl</label>
          <input id="start_url" name="start_url" type="url" placeholder="https://example.com" required>
        </div>

        <div class="field full">
          <label for="destination_url">Destination website endpoint</label>
          <input id="destination_url" name="destination_url" type="url" value="{{ default_destination_url }}" placeholder="https://your-other-site.com/api/intake" required>
          <small>The app will POST the JSON results to this URL.</small>
        </div>

        <div class="field">
          <label for="max_pages">Max pages</label>
          <input id="max_pages" name="max_pages" type="number" min="1" max="200" value="{{ max_pages_default }}">
        </div>

        <div class="field checkbox-wrap">
          <label class="checkbox-label">
            <input type="checkbox" name="restrict_to_path">
            Restrict crawl to the same path only
          </label>
        </div>

        <div class="field full">
          <label for="outbound_bearer_token">Destination bearer token</label>
          <input id="outbound_bearer_token" name="outbound_bearer_token" type="password" placeholder="Optional">
          <small>Sent as <code>Authorization: Bearer ...</code></small>
        </div>

        <div class="field full">
          <label for="webhook_secret">Webhook signing secret</label>
          <input id="webhook_secret" name="webhook_secret" type="password" placeholder="Optional">
          <small>If provided, the app adds <code>X-Signature-SHA256</code> so your receiving site can verify the payload.</small>
        </div>

        <div class="actions full">
          <button type="submit">Run Analysis</button>
        </div>
      </form>
    </section>

    <section class="card muted">
      <h2>Available endpoints</h2>
      <ul>
        <li><code>GET /health</code> - health check</li>
        <li><code>GET /api/history</code> - latest cached analyses</li>
      </ul>
    </section>
  </main>
</body>
</html>
