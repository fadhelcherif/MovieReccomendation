# 🚀 Deployment Guide - CineMatch Movie Recommendation System

## Option 1: Streamlit Community Cloud (Recommended - FREE)

### ✅ Prerequisites
- GitHub account
- Your project pushed to GitHub (✓ Already done!)

### 📋 Step-by-Step Deployment

#### 1. Sign up for Streamlit Community Cloud
1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Click **"Sign up with GitHub"**
3. Authorize Streamlit to access your GitHub repositories

#### 2. Deploy Your App
1. Click **"New app"** button
2. Fill in the deployment form:
   - **Repository**: `fadhelcherif/Movie_recc`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose your custom URL (e.g., `cinematch-recommendations`)

3. Click **"Deploy!"**

#### 3. Wait for Deployment
- Initial deployment takes 2-5 minutes
- Streamlit will:
  - Install dependencies from `requirements.txt`
  - Download your CSV files from GitHub
  - Build the TF-IDF cache
  - Launch your app

#### 4. Your App is Live! 🎉
- Access at: `https://your-app-name.streamlit.app`
- Share the link with anyone!

### ⚙️ Configuration

Your app is already configured for deployment:
- ✅ `requirements.txt` - All dependencies listed
- ✅ `.streamlit/config.toml` - Theme and settings
- ✅ `.gitignore` - Large unnecessary files excluded
- ✅ CSV files - Will be downloaded from GitHub

### 🔧 Advanced Settings (Optional)

In Streamlit Cloud dashboard, you can:
- Set **Python version** (3.8, 3.9, 3.10, 3.11)
- Add **secrets** for API keys (if needed later)
- Configure **resource limits**
- Set up **custom domain** (paid feature)

### 📊 Resource Limits (Free Tier)
- **CPU**: 1 core
- **Memory**: 1 GB RAM
- **Storage**: 1 GB
- **Bandwidth**: Unlimited
- **Uptime**: Always on (with 7-day inactivity shutdown)

### 💡 Important Notes

**File Size Considerations:**
- Your CSV files (~165 MB total) are within GitHub limits
- The cached `cosine_similarity.pkl` (482 MB) is in `.gitignore`
- App will regenerate cache on first run (takes ~2 minutes)

**Performance:**
- First load: ~2 minutes (building cache)
- Subsequent loads: ~5 seconds (using cached data)
- Recommendations: Instant (<100ms)

### 🐛 Troubleshooting

**If deployment fails:**

1. **Check logs** in Streamlit Cloud dashboard
2. **Common issues:**
   - Missing dependencies → Check `requirements.txt`
   - CSV file too large → Already handled (excluded raw dataset)
   - Memory limit → Cache is regenerated efficiently

3. **Memory optimization** (if needed):
   - Reduce `num_display` default in `app.py` (currently 40)
   - Limit TF-IDF features in `recommendation_engine.py`

### 🔄 Updating Your Deployed App

1. Make changes locally
2. Commit and push to GitHub:
   ```bash
   git add .
   git commit -m "Update: description of changes"
   git push origin main
   ```
3. Streamlit auto-deploys within 1 minute!

---

## Option 2: Heroku (Alternative - Free Tier Available)

### Setup
```bash
# Install Heroku CLI
# Download from: https://devcenter.heroku.com/articles/heroku-cli

# Login
heroku login

# Create app
heroku create cinematch-recommendations

# Deploy
git push heroku main
```

### Configuration Files Needed

**`Procfile`:**
```
web: sh setup.sh && streamlit run app.py
```

**`setup.sh`:**
```bash
mkdir -p ~/.streamlit/
echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

### Deploy
```bash
git add Procfile setup.sh
git commit -m "Add Heroku configuration"
git push heroku main
```

---

## Option 3: Railway (Modern Alternative)

1. Go to **[railway.app](https://railway.app)**
2. Connect GitHub repository
3. Select `fadhelcherif/Movie_recc`
4. Railway auto-detects Streamlit
5. Deploy automatically!

---

## Option 4: Render (Another Free Option)

1. Go to **[render.com](https://render.com)**
2. Click **"New +"** → **"Web Service"**
3. Connect GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. Deploy!

---

## 🎯 Recommended Choice

**Use Streamlit Community Cloud** because:
- ✅ **Built for Streamlit** - Zero configuration
- ✅ **Free forever** - No credit card required
- ✅ **Auto-deployment** - Push to GitHub = auto-deploy
- ✅ **Generous limits** - 1GB RAM, unlimited bandwidth
- ✅ **Easy sharing** - Beautiful streamlit.app URL
- ✅ **Analytics** - Built-in app analytics
- ✅ **Community** - Active support and examples

---

## 📧 Need Help?

1. **Streamlit Community**: [discuss.streamlit.io](https://discuss.streamlit.io)
2. **Documentation**: [docs.streamlit.io](https://docs.streamlit.io)
3. **GitHub Issues**: [Your repo issues page]

---

## 🎉 What's Next After Deployment?

1. **Share your app** - Post on social media, LinkedIn
2. **Add to portfolio** - Include in your resume/portfolio
3. **Get feedback** - Share with friends and improve
4. **Monitor usage** - Check analytics in Streamlit Cloud
5. **Iterate** - Update features based on user feedback

---

**Ready to deploy?** Follow Option 1 (Streamlit Community Cloud) above! 🚀
