# Cyberbullying Detection System

A comprehensive AI-powered cyberbullying detection system with a modern React frontend and Flask backend API.

## 🏗️ Architecture

```
Cyberbullying Detection System/
├── Frontend/                 # React TypeScript application
│   ├── src/
│   │   ├── components/      # UI components (ChatApp, Message, etc.)
│   │   ├── services/        # API communication layer
│   │   └── types/           # TypeScript type definitions
│   └── package.json
├── Backend/                 # Flask Python API
│   ├── app.py              # Main Flask application
│   ├── bullying_detector.py # AI detection logic
│   ├── config.py           # Configuration management
│   └── requirements.txt
└── package.json            # Project management scripts
```

## 🚀 Quick Start

### Prerequisites
- **Node.js** (v16 or higher)
- **Python** (3.8 or higher)
- **pip** (Python package manager)

### 1. Install Dependencies
```bash
# Install project management dependencies
npm install

# Install all dependencies for both frontend and backend
npm run setup
```

### 2. Environment Configuration
```bash
# Backend - Create .env file
cd Backend
cp .env.example .env
# Edit .env with your OpenAI API key and other settings
```

### 3. Start the Application
```bash
# Start both frontend and backend concurrently
npm start

# Or start in development mode with hot reload
npm run start:dev
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 🎯 Features

### Frontend (React + TypeScript)
- **Modern Glass Morphism UI** - Beautiful, responsive design
- **Real-time Chat Interface** - Smooth messaging experience
- **AI Safety Indicators** - Visual feedback for detection status
- **Mobile Responsive** - Works on all devices
- **TypeScript Support** - Full type safety

### Backend (Flask + AI)
- **Multi-layer Detection** - Local database + OpenAI integration
- **RESTful API** - Clean, documented endpoints
- **Batch Processing** - Handle multiple messages
- **Statistics & Analytics** - Usage monitoring
- **Configurable Thresholds** - Adjustable sensitivity

## 📡 API Endpoints

### Core Detection
- `POST /api/detect` - Analyze single message
- `POST /api/batch-detect` - Analyze multiple messages
- `GET /api/stats` - Get detection statistics
- `POST /api/add-words` - Add custom bullying terms

### Example API Usage
```javascript
// Detect bullying in a message
const response = await fetch('http://localhost:5000/api/detect', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    text: "Your message here",
    confidence_threshold: 0.7,
    include_details: true
  })
});

const result = await response.json();
console.log(result.data.is_bullying); // true/false
console.log(result.data.confidence); // 0.0 - 1.0
```

## 🛠️ Development

### Available Scripts

#### Project Level
- `npm run setup` - Install all dependencies
- `npm start` - Start both services
- `npm run start:dev` - Start with hot reload

#### Frontend Only
- `npm run start:frontend` - Start React app
- `npm run build:frontend` - Build for production
- `npm run test:frontend` - Run tests

#### Backend Only
- `npm run start:backend` - Start Flask API
- `npm run start:backend:dev` - Start with debug mode

### Environment Variables

#### Backend (.env)
```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo

# Server Configuration
HOST=0.0.0.0
PORT=5000
DEBUG=True

# Detection Settings
CONFIDENCE_THRESHOLD=0.7
LOCAL_DATABASE_PATH=data/bullying_words.json
```

#### Frontend (.env)
```env
# API Configuration
REACT_APP_API_URL=http://localhost:5000
REACT_APP_WS_URL=ws://localhost:5000
```

## 🔧 Configuration

### Detection Sensitivity
Adjust the confidence threshold in the backend configuration:
- `0.3` - Very sensitive (catches more, might have false positives)
- `0.5` - Moderate sensitivity
- `0.7` - Default (balanced)
- `0.9` - Conservative (high confidence required)

### UI Customization
Modify the frontend theme in `Frontend/src/index.css`:
- Glass morphism effects
- Color schemes
- Animations and transitions

## 📊 Monitoring & Analytics

### Real-time Statistics
- Total messages processed
- Detection accuracy rates
- Response times
- Most common flagged terms

### Logging
- All requests logged with timestamps
- Detection results and confidence scores
- Error tracking and debugging info

## 🛡️ Security Features

- **Input Validation** - Sanitize all user inputs
- **Rate Limiting** - Prevent API abuse
- **CORS Protection** - Secure cross-origin requests
- **Error Handling** - Graceful failure management

## 📱 Deployment

### Production Build
```bash
# Build frontend for production
npm run build:frontend

# The build will be in Frontend/build/
# Serve with your preferred web server
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up --build
```

### Environment-Specific Configs
- Development: Auto-reload, detailed logging
- Production: Optimized builds, minimal logging
- Testing: Mock APIs, isolated environments

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Troubleshooting

### Common Issues

**Frontend won't connect to backend:**
- Check if backend is running on port 5000
- Verify CORS configuration
- Check browser console for errors

**OpenAI API errors:**
- Verify API key in `.env` file
- Check API quota and billing
- Ensure internet connectivity

**Python dependency issues:**
- Use virtual environment: `python -m venv venv`
- Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Install: `pip install -r requirements.txt`

### Getting Help
- Create an issue in the repository
- Check existing issues for solutions
- Review logs in `Backend/bullying_detection.log`

---

**Made with ❤️ for safer online communities**
