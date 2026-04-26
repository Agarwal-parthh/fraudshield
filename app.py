from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import joblib
import numpy as np
import os

app = Flask(__name__, static_folder='static')
CORS(app)

# ── Model Definition (must match training) ──
class FraudTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, n_heads=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]
        return self.fc(x).squeeze(-1)

# ── Load everything ──
print("Loading model and assets...")
device = torch.device('cpu')

model = FraudTransformer(input_dim=5)
model.load_state_dict(torch.load('fraud_model.pt', map_location=device))
model.eval()

scaler        = joblib.load('scaler.pkl')
le_merchant   = joblib.load('le_merchant.pkl')
le_category   = joblib.load('le_category.pkl')
user_profiles = joblib.load('user_profiles.pkl')

features = ['amt', 'merchant', 'category', 'hour', 'day']

print("All assets loaded ✓")

# ── Helper functions ──
def compute_deviation(amt_scaled, hour_scaled, profile):
    amt_dev  = abs(amt_scaled - profile['avg_amt'])  / (profile['std_amt']  + 1e-5)
    time_dev = abs(hour_scaled - profile['avg_hour'])
    return float(amt_dev + time_dev)

def compute_risk(model_prob, deviation):
    return float(0.7 * model_prob + 0.3 * (deviation / (1 + deviation)))

def explain(amt_scaled, hour_scaled, profile):
    reasons = []
    if amt_scaled > profile['avg_amt'] * 2:
        reasons.append("Amount exceeds 2× user baseline")
    if abs(hour_scaled - profile['avg_hour']) > 5:
        reasons.append("Transaction at unusual hour")
    if not reasons:
        reasons.append("Behavioral pattern anomaly detected by transformer")
    return reasons

# ── Routes ──


from flask import render_template

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Get inputs
        amt      = float(data.get('amt', 1000))
        hour     = int(data.get('hour', 12))
        day      = int(data.get('day', 15))
        category = str(data.get('category', 'Shopping'))
        cc_num   = int(data.get('cc_num', 0))

        # Encode category
        try:
            cat_encoded = le_category.transform([category])[0]
        except:
            cat_encoded = 0

        # Use median merchant (0) as default
        merchant_encoded = 0

        # Build raw feature row
        raw = np.array([[amt, merchant_encoded, cat_encoded, hour, day]])
        scaled = scaler.transform(raw)[0]

        # Build sequence (repeat single txn 5 times to match seq_len=5)
        seq = np.tile(scaled, (5, 1))
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        # Model inference
        with torch.no_grad():
            logit = model(seq_tensor)
            model_prob = torch.sigmoid(logit).item()

        # User profile
        profile = user_profiles.get(cc_num, {
            'avg_amt':  scaled[0],
            'std_amt':  1.0,
            'avg_hour': scaled[3]
        })

        deviation = compute_deviation(scaled[0], scaled[3], profile)
        risk      = compute_risk(model_prob, deviation)
        reasons   = explain(scaled[0], scaled[3], profile)
        risk_pct  = min(99, round(risk * 100))

        # Classification
        if risk_pct < 30:
            verdict = 'LEGITIMATE'
        elif risk_pct < 65:
            verdict = 'SUSPICIOUS'
        else:
            verdict = 'FRAUDULENT'

        return jsonify({
            'verdict':    verdict,
            'risk_score': round(risk, 4),
            'risk_pct':   risk_pct,
            'model_prob': round(model_prob, 4),
            'deviation':  round(deviation, 4),
            'reasons':    reasons,
            'status':     'success'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'running',
        'model':   'FraudTransformer',
        'auc':     0.9564,
        'recall':  0.877
    })
@app.route('/test')
def test():
    return "WORKING"
if __name__ == '__main__':
    app.run(debug=True, port=5000)