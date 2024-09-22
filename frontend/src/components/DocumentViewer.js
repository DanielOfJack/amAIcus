import React, { useEffect, useState, useRef } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { CircularProgress, Box } from '@mui/material';
import { AiOutlineDown, AiOutlineUp } from 'react-icons/ai'; // Import arrow icons
import './DocumentViewer.css';

// Import Akoma Ntoso web components and styles
import '@lawsafrica/law-widgets/dist/components/la-akoma-ntoso';
import '@lawsafrica/law-widget-styles/css/law-widget-styles.css'; // Ensure styles are loaded

const DocumentViewer = () => {
  const { docId } = useParams();
  const location = useLocation();
  const [htmlContent, setHtmlContent] = useState(null);
  const [tocItems, setTocItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedItems, setExpandedItems] = useState({});
  const queryParams = new URLSearchParams(location.search);
  const eId = queryParams.get('eId');
  const akomaNtosoRef = useRef(null);

  useEffect(() => {
    const fetchDocument = async () => {
      try {
        const response = await fetch(`http://localhost:8000/documents/${docId}`);
        if (response.ok) {
          const data = await response.json();
          setHtmlContent(data.html_content);
          setTocItems(data.toc);
        } else {
          console.error('Error fetching document:', response.statusText);
        }
      } catch (error) {
        console.error('Error fetching document:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchDocument();
  }, [docId]);

  useEffect(() => {
    if (htmlContent && akomaNtosoRef.current) {
      akomaNtosoRef.current.innerHTML = htmlContent; // Inject the Akoma Ntoso HTML content

      if (eId) {
        const targetElement = document.getElementById(eId);
        if (targetElement) {
          targetElement.scrollIntoView({ behavior: 'smooth' });
        }
      }
    }
  }, [htmlContent, eId]);

  const toggleChildren = (id) => {
    setExpandedItems((prevExpandedItems) => ({
      ...prevExpandedItems,
      [id]: !prevExpandedItems[id]
    }));
  };

  const renderTocItems = (items) => {
    return items.map((item) => (
      <li key={item.id} className="toc-item">
        {item.children.length > 0 && (
          <button className="toc-toggle-button" onClick={() => toggleChildren(item.id)}>
            {expandedItems[item.id] ? <AiOutlineUp /> : <AiOutlineDown />}
          </button>
        )}
        <span onClick={() => scrollToSection(item.id)}>{item.title}</span>
        {item.children.length > 0 && (
          <ul
            id={`toc-children-${item.id}`}
            className="toc-children"
            style={{ display: expandedItems[item.id] ? 'block' : 'none' }}
          >
            {renderTocItems(item.children)}
          </ul>
        )}
      </li>
    ));
  };

  const scrollToSection = (sectionId) => {
    const sectionElement = document.getElementById(sectionId);
    if (sectionElement) {
      sectionElement.scrollIntoView({ behavior: 'smooth' });
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" mt={4}>
        <CircularProgress />
      </Box>
    );
  }

  if (!htmlContent) {
    return <p>Error loading document.</p>;
  }

  return (
    <div className="document-viewer">
      {/* Navbar */}
      <div className="toc-navbar">Table of Contents</div>
      {/* Back to Search Results Button */}
      <a href="/" className="back-link">Back to Search Results</a>
      
      {/* TOC Sidebar */}
      <div className="toc-sidebar">
        <ul className="toc-list">
          {renderTocItems(tocItems)}
        </ul>
      </div>

      {/* Document Content */}
      <div className="document-content">
        <la-akoma-ntoso ref={akomaNtosoRef}></la-akoma-ntoso>
      </div>
    </div>
  );
};

export default DocumentViewer;