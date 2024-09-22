// src/App.js
import React, { useState } from 'react';
import Sidebar from './components/Sidebar';
import SearchBar from './components/SearchBar';
import SearchResultsContainer from './components/SearchResultsContainer';
import DocumentViewer from './components/DocumentViewer';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';

const theme = createTheme({
  palette: {
    primary: { main: '#000000' }, // Black
    secondary: { main: '#ffffff' }, // White
    text: { primary: '#000000', secondary: '#ffffff' }, // Black and white text
    background: { default: '#ffffff' }, // White background
    action: { active: '#000000' }, // Black active color
  },
});

function App() {
  const [documents, setDocuments] = useState([]);  // State to hold the documents

  const handleSearchResults = (results) => {
    setDocuments(results);
  };

  return (
    <ThemeProvider theme={theme}>
      <Router>
        <MainContent handleSearchResults={handleSearchResults} documents={documents} />
      </Router>
    </ThemeProvider>
  );
}

function MainContent({ handleSearchResults, documents }) {
  const location = useLocation();

  // Determine if we are on the document viewer page
  const isDocumentViewer = location.pathname.startsWith('/document/');

  return (
    <div style={{ display: 'flex' }}>
      {/* Conditionally render the Sidebar */}
      {!isDocumentViewer && <Sidebar />}
      <main style={{ marginLeft: isDocumentViewer ? '0' : '340px', paddingTop: '0px', width: '100%' }}>
        <Routes>
          <Route
            path="/"
            element={
              <>
                <SearchBar onSearchResults={handleSearchResults} />
                <SearchResultsContainer documents={documents} />
              </>
            }
          />
          <Route path="/document/:docId" element={<DocumentViewer />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;