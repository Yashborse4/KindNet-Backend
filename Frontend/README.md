# KindNet - Frontend

🌟 **A Modern Cyberbullying Detection Frontend Interface**

A beautiful, modern React chat application with glass morphism design for cyber bullying detection. This frontend provides a responsive and elegant interface for real-time chat with AI-powered safety monitoring.

## Features

- 🎨 **Modern Glass Morphism Design** - Beautiful, translucent UI with rounded corners
- 📱 **Fully Responsive** - Works seamlessly on desktop, tablet, and mobile devices
- 🤖 **AI Safety Integration** - Ready for API integration with cyber bullying detection services
- ⚡ **Real-time Chat** - Smooth message flow with typing indicators
- 🛡️ **Safety Controls** - Toggle detection on/off with visual feedback
- 🎯 **TypeScript Support** - Full type safety and better development experience
- 🌈 **Tailwind CSS** - Modern utility-first styling with custom animations

## Tech Stack

- **React 18** with TypeScript
- **Tailwind CSS** for styling
- **Lucide React** for icons
- **Custom Glass Morphism** effects

## Getting Started

### Prerequisites

- Node.js (version 16 or higher)
- npm or yarn

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start the development server:**
   ```bash
   npm start
   ```

3. **Open your browser:**
   Navigate to `http://localhost:3000`

### Build for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── ChatApp.tsx        # Main chat application component
│   ├── ChatHeader.tsx     # Header with controls and status
│   ├── Message.tsx        # Individual message component
│   └── MessageInput.tsx   # Input field with send functionality
├── App.tsx               # Main app component
├── index.tsx            # React entry point
└── index.css           # Global styles and Tailwind config
```

Built with React TypeScript and Tailwind CSS, KindNet Frontend provides a clean, intuitive interface for real-time cyberbullying detection and content moderation.

## 🚀 Features

- **Real-time Text Analysis**: Instant cyberbullying detection as you type
- **Modern UI/UX**: Clean, responsive design built with Tailwind CSS
- **TypeScript Support**: Full type safety and enhanced developer experience
- **Multi-language Detection**: Support for various languages and contexts
- **Confidence Scoring**: Visual feedback with confidence levels
- **Context Awareness**: Advanced analysis considering conversation context

## 🛠️ Tech Stack

- **Framework**: React 18 with TypeScript
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **Build Tool**: Create React App
- **HTTP Client**: Fetch API
- **Development**: Hot reload, ESLint, Prettier

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Yashborse4/KindNet-Frontend.git
cd KindNet-Frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 🔧 Environment Setup

1. Copy the environment example file:
```bash
cp .env.example .env
```

2. Configure your backend API endpoint:
```env
REACT_APP_API_URL=http://localhost:5000
```

## 🎨 Available Scripts

- `npm start` - Start development server
- `npm run build` - Build for production
- `npm test` - Run test suite
- `npm run eject` - Eject from CRA (irreversible)

## 🔗 Backend Integration

This frontend connects to the KindNet Backend API. Make sure the backend is running on the configured endpoint.

**Backend Repository**: [KindNet-Backend](https://github.com/Yashborse4/KindNet-Backend)

## 📱 Component Structure

```
src/
├── components/          # React components
│   ├── ChatApp.tsx     # Main chat interface
│   ├── ChatHeader.tsx  # Header component
│   ├── Message.tsx     # Message display
│   └── MessageInput.tsx # Input component
├── services/           # API services
├── types/             # TypeScript definitions
└── utils/            # Utility functions
```

## 🎯 Usage

1. **Text Input**: Type or paste text in the input field
2. **Real-time Analysis**: See instant feedback on potential cyberbullying content
3. **Confidence Levels**: Visual indicators show detection confidence
4. **Context Understanding**: Advanced AI considers conversation context
5. **Multi-language**: Supports detection in multiple languages

## 🔒 Features

### Text Analysis
- Real-time cyberbullying detection
- Sentiment analysis
- Threat level assessment
- Context-aware filtering

### User Interface
- Responsive design for all devices
- Dark/light mode support
- Accessibility features
- Smooth animations and transitions

### Performance
- Optimized bundle size
- Lazy loading components
- Efficient API calls
- Caching strategies

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Make your changes and commit: `git commit -m 'Add new feature'`
4. Push to the branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/Yashborse4/KindNet-Frontend/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🎉 Acknowledgments

- React and TypeScript communities
- Tailwind CSS for the beautiful styling system
- All contributors and testers

---

**Made with ❤️ for a safer internet**