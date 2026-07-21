"""
House Price Estimator - Flask backend (self-contained, single file)
---------------------------------------------------------------------
Loads a pre-trained scikit-learn LinearRegression model (linear_model.pkl)
and serves a web UI + JSON prediction endpoint. The HTML/CSS/JS are
inlined below so this deploys with ONLY three files:

    app.py, requirements.txt, linear_model.pkl

(a Procfile is recommended too, but Render can also auto-detect
"gunicorn app:app" if you set it as the Start Command manually).

Local run:      python app.py
Production run: gunicorn app:app
"""

import os
import warnings
import pickle

import numpy as np
from flask import Flask, request, jsonify, render_template_string

warnings.filterwarnings("ignore")  # silence sklearn version-mismatch warning

app = Flask(__name__)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear_model.pkl")

# Change this if your model's target is in a different currency.
CURRENCY_SYMBOL = "$"

# Exact order of features the model was trained on. Do NOT reorder.
FEATURE_ORDER = [
    "bedrooms", "bathrooms", "living_area", "lot_area", "floors",
    "waterfront", "views", "condition", "grade", "house_area_excl_basement",
    "basement_area", "built_year", "renovation_year", "lot_area_renov",
    "schools_nearby", "airport_distance",
]

FIELD_LABELS = {
    "bedrooms": "Number of Bedrooms",
    "bathrooms": "Number of Bathrooms",
    "living_area": "Living Area (sqft)",
    "lot_area": "Lot Area (sqft)",
    "floors": "Number of Floors",
    "waterfront": "Waterfront Present",
    "views": "Number of Views",
    "condition": "Condition of the House",
    "grade": "Grade of the House",
    "house_area_excl_basement": "House Area Excl. Basement (sqft)",
    "basement_area": "Basement Area (sqft)",
    "built_year": "Built Year",
    "renovation_year": "Renovation Year",
    "lot_area_renov": "Renovated Lot Area (sqft)",
    "schools_nearby": "Schools Nearby",
    "airport_distance": "Distance from Airport (km)",
}

# --------------------------------------------------------------------------
# Load model once at startup
# --------------------------------------------------------------------------
model = None
model_load_error = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as exc:  # noqa: BLE001
    model_load_error = str(exc)

# --------------------------------------------------------------------------
# Inline template (HTML + CSS + JS in one string -- no templates/ or
# static/ folders required)
# --------------------------------------------------------------------------
PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Estate Value — Instant Home Price Estimator</title>
<meta name="description" content="Get an instant, model-based estimate of your home's market value." />

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;0,9..144,700;1,9..144,500&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
:root {
  --ink-navy: #0b1224;
  --deep-slate: #131b32;
  --panel: rgba(19, 27, 50, 0.55);
  --panel-border: rgba(255, 255, 255, 0.07);
  --foil-gold: #d4af6a;
  --foil-gold-soft: rgba(212, 175, 106, 0.35);
  --copper: #e8935b;
  --mist: #edeff5;
  --slate-blue: #7c8aa8;
  --error: #f0768a;
  --font-display: "Fraunces", Georgia, serif;
  --font-body: "Inter", -apple-system, BlinkMacSystemFont, sans-serif;
  --font-mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
  --shadow-card: 0 1px 1px rgba(0,0,0,.25), 0 10px 24px -6px rgba(0,0,0,.45), 0 32px 64px -16px rgba(0,0,0,.55);
  --shadow-cert: 0 2px 2px rgba(0,0,0,.3), 0 24px 48px -12px rgba(0,0,0,.6), 0 0 0 1px rgba(212,175,106,.22) inset, 0 0 70px -12px rgba(212,175,106,.4);
  --shadow-btn: 0 1px 1px rgba(0,0,0,.25), 0 10px 20px -6px rgba(0,0,0,.5), 0 0 36px -6px rgba(212,175,106,.55);
  --shadow-btn-hover: 0 1px 1px rgba(0,0,0,.25), 0 16px 28px -6px rgba(0,0,0,.55), 0 0 48px -4px rgba(212,175,106,.75);
}
* { box-sizing: border-box; }
html, body { margin:0; padding:0; background: linear-gradient(165deg,#0b1224 0%,#0d1730 55%,#131b32 100%); color: var(--mist); font-family: var(--font-body); min-height:100%; }
:focus-visible { outline: 2px solid var(--foil-gold); outline-offset: 2px; border-radius: 4px; }
.blueprint-grid { position:fixed; inset:0; z-index:-2; pointer-events:none;
  background-image:
    repeating-linear-gradient(0deg, rgba(124,138,168,.07) 0 1px, transparent 1px 42px),
    repeating-linear-gradient(90deg, rgba(124,138,168,.07) 0 1px, transparent 1px 42px),
    radial-gradient(circle at 82% 8%, rgba(212,175,106,.14), transparent 45%),
    radial-gradient(circle at 6% 92%, rgba(232,147,91,.10), transparent 45%);
}
.vignette { position:fixed; inset:0; z-index:-1; pointer-events:none; background: radial-gradient(ellipse at 50% 0%, transparent 40%, rgba(11,18,36,.75) 100%); }
.page { max-width:1180px; margin:0 auto; padding:64px 24px 32px; }
.hero { text-align:center; max-width:640px; margin:0 auto 48px; animation: rise .7s ease both; }
.kicker { font-family: var(--font-mono); font-size:12px; letter-spacing:.18em; color: var(--foil-gold); margin:0 0 18px; }
.hero-title { font-family: var(--font-display); font-weight:600; font-size: clamp(2.1rem,5vw,3.2rem); line-height:1.12; margin:0 0 18px; color: var(--mist); }
.hero-title .accent { font-style: italic; background: linear-gradient(100deg, var(--foil-gold), var(--copper) 60%); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { font-size:15.5px; line-height:1.6; color: var(--slate-blue); margin:0; }
.layout { display:grid; grid-template-columns: 1.55fr 1fr; gap:32px; align-items:start; }
.card { background: var(--panel); border:1px solid var(--panel-border); border-radius:20px; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px); box-shadow: var(--shadow-card); animation: rise .7s ease both; }
.form-card { padding:40px; }
.field-group { border:none; margin:0 0 30px; padding:0 0 30px; border-bottom:1px solid rgba(255,255,255,.07); }
.field-group:last-of-type { border-bottom:none; padding-bottom:8px; margin-bottom:28px; }
.field-group legend { display:flex; align-items:center; gap:10px; padding:0; margin-bottom:20px; font-family: var(--font-body); font-weight:600; font-size:15px; color: var(--mist); }
.eyebrow { font-family: var(--font-mono); font-size:11px; letter-spacing:.06em; color: var(--foil-gold); background: rgba(212,175,106,.12); border:1px solid rgba(212,175,106,.3); border-radius:999px; padding:3px 9px; }
.field-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:18px 20px; }
.field label { display:block; font-size:12.5px; font-weight:500; color: var(--slate-blue); margin-bottom:7px; }
.field input, .field select { width:100%; background: rgba(255,255,255,.03); border:1px solid rgba(255,255,255,.1); border-radius:10px; padding:11px 12px; color: var(--mist); font-family: var(--font-body); font-size:14px; transition: border-color .2s ease, box-shadow .2s ease, background .2s ease; }
.field select option { background: var(--deep-slate); color: var(--mist); }
.field input:hover, .field select:hover { border-color: rgba(255,255,255,.2); }
.field input:focus, .field select:focus { outline:none; border-color: var(--foil-gold); box-shadow: 0 0 0 3px rgba(212,175,106,.16); background: rgba(255,255,255,.05); }
.submit-btn { position:relative; width:100%; border:none; border-radius:12px; padding:16px 28px; margin-top:4px; background: linear-gradient(135deg, var(--foil-gold), var(--copper)); color:#201202; font-family: var(--font-body); font-weight:700; font-size:15px; letter-spacing:.01em; cursor:pointer; box-shadow: var(--shadow-btn); transition: transform .2s ease, box-shadow .2s ease, opacity .2s ease; }
.submit-btn:hover { transform: translateY(-2px); box-shadow: var(--shadow-btn-hover); }
.submit-btn:active { transform: translateY(0); }
.submit-btn:disabled { opacity:.72; cursor:not-allowed; transform:none; }
.btn-spinner { display:none; width:16px; height:16px; margin-left:10px; border-radius:50%; border:2px solid rgba(32,18,2,.35); border-top-color:#201202; vertical-align:-3px; animation: spin .7s linear infinite; }
.submit-btn.loading .btn-spinner { display:inline-block; }
.submit-btn.loading .btn-label { opacity:.75; }
.form-error { margin:16px 0 0; padding:11px 14px; border-radius:10px; background: rgba(240,118,138,.08); border:1px solid rgba(240,118,138,.3); color: var(--error); font-size:13px; line-height:1.5; }
.certificate { position:sticky; top:32px; padding:40px 32px; text-align:center; background: linear-gradient(160deg, rgba(212,175,106,.07), rgba(19,27,50,.88)); border:1px solid var(--foil-gold-soft); box-shadow: var(--shadow-cert); overflow:hidden; }
.certificate-corner { position:absolute; width:20px; height:20px; border:2px solid var(--foil-gold); opacity:.65; }
.certificate-corner.tl { top:12px; left:12px; border-right:none; border-bottom:none; border-radius:4px 0 0 0; }
.certificate-corner.tr { top:12px; right:12px; border-left:none; border-bottom:none; border-radius:0 4px 0 0; }
.certificate-corner.bl { bottom:12px; left:12px; border-right:none; border-top:none; border-radius:0 0 0 4px; }
.certificate-corner.br { bottom:12px; right:12px; border-left:none; border-top:none; border-radius:0 0 4px 0; }
.cert-eyebrow { font-family: var(--font-mono); text-transform:uppercase; letter-spacing:.16em; font-size:12px; color: var(--foil-gold); margin:0 0 6px; }
.cert-sub { font-size:13.5px; color: var(--slate-blue); margin:0; }
.cert-value-wrap { display:flex; align-items:baseline; justify-content:center; gap:6px; margin:30px 0 14px; min-height:64px; }
.cert-currency { font-family: var(--font-display); font-size:1.7rem; color: var(--foil-gold); }
.cert-value { font-family: var(--font-display); font-weight:600; font-size: clamp(2.2rem,5vw,3.1rem); letter-spacing:-.01em; color: var(--mist); font-variant-numeric: tabular-nums; }
.certificate.revealed .cert-value { animation: shimmer-in 1s ease both; }
.cert-hint { font-size:13px; color: var(--slate-blue); line-height:1.55; margin:0; }
.cert-divider { height:1px; margin:28px 0; background: linear-gradient(90deg, transparent, rgba(212,175,106,.45), transparent); }
.cert-meta { margin:0; display:flex; flex-direction:column; gap:12px; }
.cert-meta > div { display:flex; align-items:baseline; justify-content:space-between; gap:12px; }
.cert-meta dt { font-family: var(--font-mono); text-transform:uppercase; letter-spacing:.06em; font-size:10.5px; color: var(--slate-blue); margin:0; }
.cert-meta dd { font-size:13.5px; font-weight:500; color: var(--mist); margin:0; text-align:right; }
.cert-disclaimer { margin:26px 0 0; font-size:11px; line-height:1.6; color: var(--slate-blue); opacity:.8; }
.page-footer { text-align:center; padding:44px 0 12px; font-family: var(--font-mono); font-size:12px; letter-spacing:.03em; color: var(--slate-blue); opacity:.65; }
@keyframes rise { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:translateY(0);} }
@keyframes spin { to{ transform:rotate(360deg); } }
@keyframes shimmer-in { 0%{opacity:0; transform:translateY(6px) scale(.98);} 60%{opacity:1;} 100%{opacity:1; transform:translateY(0) scale(1);} }
@media (prefers-reduced-motion: reduce) { * { animation-duration:.001ms !important; animation-iteration-count:1 !important; transition-duration:.001ms !important; } }
@media (max-width:900px){ .layout{ grid-template-columns:1fr; } .certificate{ position:static; } .page{ padding:48px 18px 24px; } .form-card{ padding:28px 22px; } }
@media (max-width:480px){ .field-grid{ grid-template-columns:1fr; } }
</style>
</head>
<body>

<div class="blueprint-grid" aria-hidden="true"></div>
<div class="vignette" aria-hidden="true"></div>

<div class="page">

  <header class="hero">
    <p class="kicker">INSTANT HOME VALUATION</p>
    <h1 class="hero-title">What is your home<br />actually <span class="accent">worth?</span></h1>
    <p class="hero-sub">Enter your property's details below. A linear regression model trained on 16 real
      property factors will appraise it in an instant — no agent, no waiting.</p>
  </header>

  <main class="layout">

    <form id="valuation-form" class="card form-card" autocomplete="off" novalidate>

      <fieldset class="field-group">
        <legend><span class="eyebrow">01</span> Basics</legend>
        <div class="field-grid">
          <div class="field"><label for="bedrooms">Bedrooms</label>
            <input type="number" id="bedrooms" name="bedrooms" min="0" max="15" step="1" value="3" required /></div>
          <div class="field"><label for="bathrooms">Bathrooms</label>
            <input type="number" id="bathrooms" name="bathrooms" min="0" max="10" step="0.25" value="2" required /></div>
          <div class="field"><label for="floors">Floors</label>
            <input type="number" id="floors" name="floors" min="1" max="4" step="0.5" value="1" required /></div>
          <div class="field"><label for="waterfront">Waterfront present</label>
            <select id="waterfront" name="waterfront" required>
              <option value="0" selected>No</option>
              <option value="1">Yes</option>
            </select></div>
        </div>
      </fieldset>

      <fieldset class="field-group">
        <legend><span class="eyebrow">02</span> Space &amp; structure</legend>
        <div class="field-grid">
          <div class="field"><label for="living_area">Living area (sqft)</label>
            <input type="number" id="living_area" name="living_area" min="100" step="10" value="1800" required /></div>
          <div class="field"><label for="lot_area">Lot area (sqft)</label>
            <input type="number" id="lot_area" name="lot_area" min="100" step="10" value="5000" required /></div>
          <div class="field"><label for="house_area_excl_basement">House area excl. basement (sqft)</label>
            <input type="number" id="house_area_excl_basement" name="house_area_excl_basement" min="0" step="10" value="1800" required /></div>
          <div class="field"><label for="basement_area">Basement area (sqft)</label>
            <input type="number" id="basement_area" name="basement_area" min="0" step="10" value="0" required /></div>
          <div class="field"><label for="grade">Grade of the house (1–13)</label>
            <input type="number" id="grade" name="grade" min="1" max="13" step="1" value="7" required /></div>
          <div class="field"><label for="condition">Condition (1–5)</label>
            <select id="condition" name="condition" required>
              <option value="1">1 — Poor</option>
              <option value="2">2 — Fair</option>
              <option value="3" selected>3 — Average</option>
              <option value="4">4 — Good</option>
              <option value="5">5 — Excellent</option>
            </select></div>
        </div>
      </fieldset>

      <fieldset class="field-group">
        <legend><span class="eyebrow">03</span> Amenities &amp; location</legend>
        <div class="field-grid">
          <div class="field"><label for="views">Number of views (0–4)</label>
            <input type="number" id="views" name="views" min="0" max="4" step="1" value="0" required /></div>
          <div class="field"><label for="schools_nearby">Schools nearby</label>
            <input type="number" id="schools_nearby" name="schools_nearby" min="0" max="20" step="1" value="2" required /></div>
          <div class="field"><label for="airport_distance">Distance from airport (km)</label>
            <input type="number" id="airport_distance" name="airport_distance" min="0" step="0.5" value="15" required /></div>
        </div>
      </fieldset>

      <fieldset class="field-group">
        <legend><span class="eyebrow">04</span> History</legend>
        <div class="field-grid">
          <div class="field"><label for="built_year">Built year</label>
            <input type="number" id="built_year" name="built_year" min="1800" max="2026" step="1" value="2005" required /></div>
          <div class="field"><label for="renovation_year">Renovation year (0 if never)</label>
            <input type="number" id="renovation_year" name="renovation_year" min="0" max="2026" step="1" value="0" required /></div>
          <div class="field"><label for="lot_area_renov">Renovated lot area (sqft)</label>
            <input type="number" id="lot_area_renov" name="lot_area_renov" min="0" step="10" value="5000" required /></div>
        </div>
      </fieldset>

      <button type="submit" id="submit-btn" class="submit-btn">
        <span class="btn-label">Estimate value</span>
        <span class="btn-spinner" aria-hidden="true"></span>
      </button>

      <p id="form-error" class="form-error" role="alert" hidden></p>
    </form>

    <aside class="card certificate" id="certificate">
      <div class="certificate-corner tl" aria-hidden="true"></div>
      <div class="certificate-corner tr" aria-hidden="true"></div>
      <div class="certificate-corner bl" aria-hidden="true"></div>
      <div class="certificate-corner br" aria-hidden="true"></div>

      <p class="cert-eyebrow">Appraisal Certificate</p>
      <p class="cert-sub">Estimated market value</p>

      <div class="cert-value-wrap">
        <span class="cert-currency">{{ currency }}</span>
        <span class="cert-value" id="cert-value">—</span>
      </div>

      <p class="cert-hint" id="cert-hint">Fill in the form and press "Estimate value" to appraise this property.</p>

      <div class="cert-divider" aria-hidden="true"></div>

      <dl class="cert-meta">
        <div><dt>Factors weighed</dt><dd>16 property attributes</dd></div>
        <div><dt>Model</dt><dd>Linear Regression</dd></div>
        <div><dt>Generated</dt><dd id="cert-timestamp">—</dd></div>
      </dl>

      <p class="cert-disclaimer">This is a statistical estimate for reference only, not a certified appraisal.</p>
    </aside>

  </main>

  <footer class="page-footer">
    <p>Built with a scikit-learn linear regression model &middot; served with Flask</p>
  </footer>

</div>

<script>
(function () {
  "use strict";
  var form = document.getElementById("valuation-form");
  var submitBtn = document.getElementById("submit-btn");
  var errorBox = document.getElementById("form-error");
  var certificate = document.getElementById("certificate");
  var certValue = document.getElementById("cert-value");
  var certHint = document.getElementById("cert-hint");
  var certTimestamp = document.getElementById("cert-timestamp");

  var FIELD_IDS = ["bedrooms","bathrooms","living_area","lot_area","floors",
    "waterfront","views","condition","grade","house_area_excl_basement",
    "basement_area","built_year","renovation_year","lot_area_renov",
    "schools_nearby","airport_distance"];

  function showError(message) { errorBox.textContent = message; errorBox.hidden = false; }
  function hideError() { errorBox.hidden = true; errorBox.textContent = ""; }
  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    submitBtn.classList.toggle("loading", isLoading);
  }

  function animateValueTo(target) {
    var start = 0, duration = 900, startTime = performance.now();
    function frame(now) {
      var progress = Math.min((now - startTime) / duration, 1);
      var eased = 1 - Math.pow(1 - progress, 3);
      var current = Math.round(start + (target - start) * eased);
      certValue.textContent = current.toLocaleString("en-US");
      if (progress < 1) { requestAnimationFrame(frame); }
      else { certValue.textContent = target.toLocaleString("en-US", { maximumFractionDigits: 0 }); }
    }
    requestAnimationFrame(frame);
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    hideError();

    var payload = {};
    FIELD_IDS.forEach(function (id) {
      payload[id] = document.getElementById(id).value;
    });

    setLoading(true);

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    })
      .then(function (response) {
        return response.json().then(function (data) {
          if (!response.ok || !data.success) {
            throw new Error(data.error || "Something went wrong. Please try again.");
          }
          return data;
        });
      })
      .then(function (data) {
        certificate.classList.remove("revealed");
        void certificate.offsetWidth;
        certificate.classList.add("revealed");
        animateValueTo(Math.round(data.prediction));
        certHint.textContent = "Estimate generated from the 16 details you provided.";
        var now = new Date();
        certTimestamp.textContent = now.toLocaleString("en-US", {
          month: "short", day: "numeric", year: "numeric", hour: "numeric", minute: "2-digit",
        });
      })
      .catch(function (err) {
        showError(err.message || "Unable to reach the prediction service.");
      })
      .finally(function () {
        setLoading(false);
      });
  });
})();
</script>
</body>
</html>
"""

# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE_TEMPLATE, currency=CURRENCY_SYMBOL)


@app.route("/health")
def health():
    status = "ok" if model is not None else "model_not_loaded"
    return jsonify({"status": status}), (200 if model is not None else 500)


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "success": False,
            "error": f"Model failed to load on the server: {model_load_error}",
        }), 500

    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({"success": False, "error": "No input data received."}), 400

    values = []
    for key in FEATURE_ORDER:
        raw_value = payload.get(key, "")
        if raw_value in ("", None):
            label = FIELD_LABELS.get(key, key)
            return jsonify({"success": False, "error": f"'{label}' is required."}), 400
        try:
            values.append(float(raw_value))
        except (TypeError, ValueError):
            label = FIELD_LABELS.get(key, key)
            return jsonify({"success": False, "error": f"'{label}' must be a number."}), 400

    try:
        features = np.array(values, dtype=float).reshape(1, -1)
        raw_prediction = model.predict(features)[0]
        prediction = round(max(float(raw_prediction), 0), 2)
        return jsonify({"success": True, "prediction": prediction})
    except Exception as exc:  # noqa: BLE001
        return jsonify({
            "success": False,
            "error": f"The model could not generate a prediction: {exc}",
        }), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"success": False, "error": "Route not found."}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({"success": False, "error": "Internal server error."}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
