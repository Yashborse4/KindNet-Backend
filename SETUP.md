# 🚀 Cyberbullying Detection System - Setup Guide

This guide will walk you through setting up and running the complete cyberbullying detection system.

## 📋 Prerequisites

Before starting, ensure you have the following installed:

- **Node.js** (version 16 or higher) - [Download here](https://nodejs.org/)
- **Python** (version 3.8 or higher) - [Download here](https://python.org/)
- **Git** (optional, for cloning) - [Download here](https://git-scm.com/)

## 🔧 Quick Setup

### Option 1: Automated Setup (Recommended)

1. **Install project dependencies:**
   ```bash
   npm install
   npm run setup
   ```

2. **Configure environment:**
   ```bash
   # Backend configuration
   cd Backend
   copy .env.example .env
   # Edit .env file with your OpenAI API key
   notepad .env
   cd ..
   ```

3. **Start the application:**
   ```bash
   # Start both frontend and backend
   npm start
   
   # OR use the startup scripts
   # Windows: Double-click start.bat or run start.ps1
   # The application will open automatically in your browser
   ```

### Option 2: Manual Setup

1. **Setup Backend:**
   ```bash
   cd Backend
   pip install -r requirements.txt
   copy .env.example .env
   # Edit .env with your configuration
   python app.py
   ```

2. **Setup Frontend (in a new terminal):**
   ```bash
   cd Frontend
   npm install
   npm start
   ```

## ⚙️ Configuration

### Backend Configuration (Backend/.env)

```env
# OpenAI API Key (required for AI detection)
OPENAI_API_KEY=your_openai_api_key_here

# Server settings
HOST=0.0.0.0
PORT=5000
DEBUG=True

# Detection settings
CONFIDENCE_THRESHOLD=0.7
LOCAL_DATABASE_PATH=data/bullying_words.json

# CORS settings
CORS_ORIGINS=http://localhost:3000
```

### Frontend Configuration (Frontend/.env)

```env
# API endpoint
REACT_APP_API_URL=http://localhost:5000
REACT_APP_API_TIMEOUT=30000

# Feature flags
REACT_APP_ENABLE_VOICE_MESSAGES=true
REACT_APP_ENABLE_FILE_UPLOAD=true
```

## 🧪 Testing the Setup

### 1. Run Integration Tests

```bash
# Install test dependencies
npm install node-fetch

# Run integration tests (backend must be running)
node test-integration.js
```

### 2. Manual Testing

1. **Backend Health Check:**
   - Open http://localhost:5000
   - Should return: `{"status": "healthy", ...}`

2. **Test API Endpoints:**
   ```bash
   # Test detection endpoint
   curl -X POST http://localhost:5000/api/detect \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}'
   ```

3. **Frontend Testing:**
   - Open http://localhost:3000
   - Should show the glass morphism chat interface
   - Try sending various messages to test detection

## 📱 Usage Instructions

### Chat Interface Features

1. **Send Messages:** Type in the input field and press Enter
2. **Toggle Detection:** Click the shield icon to enable/disable AI monitoring
3. **View Results:** Each message shows detection results and confidence scores
4. **Responsive Design:** Works on desktop, tablet, and mobile devices

### API Usage Examples

```javascript
// Detect bullying in a message
const response = await fetch('http://localhost:5000/api/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: "Your message here",
    confidence_threshold: 0.7,
    include_details: true
  })
});

const result = await response.json();
console.log('Is bullying:', result.data.is_bullying);
console.log('Confidence:', result.data.confidence);
```

## 🔍 Troubleshooting

### Common Issues

**"Backend not responding"**
- Check if backend is running on port 5000
- Verify no firewall blocking
- Check console for error messages

**"OpenAI API errors"**
- Verify your API key in Backend/.env
- Check your OpenAI account has credits
- Ensure internet connectivity

**"Frontend won't start"**
- Clear npm cache: `npm cache clean --force`
- Delete node_modules and reinstall: `rm -rf node_modules && npm install`
- Check for port conflicts (port 3000)

**"Detection not working"**
- Check backend logs for errors
- Verify .env configuration
- Test with simple messages first

### Getting Help

1. Check the logs:
   - Backend: `Backend/bullying_detection.log`
   - Frontend: Browser developer console

2. Test individual components:
   - Backend: `cd Backend && python test_api.py`
   - Frontend: `cd Frontend && npm test`

3. Verify network connectivity:
   - Backend health: http://localhost:5000
   - Frontend health: http://localhost:3000

## 🌟 Advanced Configuration

### Custom Detection Rules

Add custom bullying words to `Backend/data/bullying_words.json`:
```json
{
  "harmful_words": ["word1", "word2"],
  "categories": {
    "harassment": ["bully", "harass"],
    "threats": ["hurt", "harm"]
  }
}
```

### Performance Tuning

1. **Increase confidence threshold** for fewer false positives
2. **Adjust API timeout** for slower connections
3. **Enable caching** for repeated requests
4. **Use batch detection** for multiple messages

### Production Deployment

```bash
# Build frontend for production
cd Frontend && npm run build

# Run backend in production mode
cd Backend && python app.py

# Use Docker for containerized deployment
docker-compose up --build
```

## 📚 Next Steps

1. **Customize the UI:** Modify colors, themes, and layouts in `Frontend/src/index.css`
2. **Add Features:** Implement file upload, voice messages, or user authentication
3. **Integrate:** Connect with your existing chat system or social platform
4. **Monitor:** Set up logging, analytics, and performance monitoring
5. **Scale:** Deploy to cloud services for production use

## 🆘 Support

- **Documentation:** See README.md for detailed information
- **Issues:** Create GitHub issues for bugs or feature requests
- **Community:** Join our discussions for help and sharing

---

**🎉 Congratulations! Your cyberbullying detection system is now ready to help create safer online communities!**
