// src/components/SearchResultCard.js
import React, { useState, useRef } from 'react';
import { Box, Typography, Divider, IconButton, Link } from '@mui/material';
import MessageIcon from '@mui/icons-material/Message';
import CloseIcon from '@mui/icons-material/Close';
import { useNavigate } from 'react-router-dom';
import './SearchResultCard.css';

// Function to map country codes to country names
const getCountryName = (countryCode) => {
  const countryMap = {
    za: 'South Africa',
    zw: 'Zimbabwe',
    ng: 'Nigeria',
    ke: 'Kenya',
    // Add more country codes as necessary
  };
  return countryMap[countryCode.toLowerCase()] || countryCode;
};

// Mapping for location codes
const getLocationName = (locationCode) => {
  const locationMap = {
    cpt: 'Cape Town',
    jhb: 'Johannesburg',
    wc033: 'Cape Agulhas',
    wc011: 'Matzikama',
    eth: 'eThekwini',
    ec443: 'Mbizana',
    // Add more locations as necessary
  };
  return locationMap[locationCode?.toLowerCase()] || locationCode || '';
};

// Adjust the subtype formatting (Capitalize first letter regardless of the value)
const formatSubtype = (subtype) => {
  return subtype.charAt(0).toUpperCase() + subtype.slice(1);
};

// Function to format a component's title
const formatComponentTitle = (component) => {
  const { tag, num, heading, subheading, crossheading } = component;

  const capitalizedTag = tag ? tag.charAt(0).toUpperCase() + tag.slice(1) : '';
  const formattedNum = num && num !== "N/A" ? (num.endsWith('.') ? num.slice(0, -1) : num) : '';
  
  let titleParts = [];
  if (capitalizedTag && formattedNum) {
    titleParts.push(`${capitalizedTag} ${formattedNum}`);
  } else if (capitalizedTag) {
    titleParts.push(`${capitalizedTag}`);
  }

  if (heading && heading !== "N/A") {
    titleParts.push(heading);
  }
  if (subheading && subheading !== "N/A") {
    titleParts.push(subheading);
  }
  if (crossheading && crossheading !== "N/A") {
    titleParts.push(crossheading);
  }

  return titleParts.join(' | ');
};

const SearchResultCard = ({ documentTitle, countryName, flagUrl, documentType, date, searchResults, docId }) => {
  const [activePopupIndex, setActivePopupIndex] = useState(null);
  const [popupPosition, setPopupPosition] = useState({ top: 0 });

  const buttonRefs = useRef([]); // Store references to all message buttons
  const navigate = useNavigate();

  const togglePopup = (index) => {
    if (activePopupIndex === index) {
      setActivePopupIndex(null);
    } else {
      const button = buttonRefs.current[index];
      const buttonRect = button.getBoundingClientRect();
      const popupTop = buttonRect.top + window.scrollY + buttonRect.height / 2;
      setPopupPosition({ top: popupTop });
      setActivePopupIndex(index);
    }
  };

  // Extract location and country from the country code (e.g., za-wc011)
  const locationCode = countryName.split('-')[1];
  const countryCode = countryName.split('-')[0];
  const locationName = getLocationName(locationCode);
  const fullCountryName = getCountryName(countryCode);

  // Find the highest score from the search results
  const highestScore = Math.max(...searchResults.map((result) => result.score));

  // Initialize previousComponents to keep track of components from the previous hit
  let previousComponents = [];

  return (
    <Box className="search-result-card">
      {/* Document Title with Date */}
      <Box className="document-info">
        <Typography variant="h5" className="document-title">
          {documentTitle}, {date}
        </Typography>
        {/* Display the highest score in the top right corner */}
        <Typography className="highest-score" style={{ float: 'right' }}>
          Highest Score: {highestScore.toFixed(2)}
        </Typography>
      </Box>

      {/* Country Info Row */}
      <Box className="country-info-row">
        <img src={flagUrl} alt={`Flag of ${fullCountryName}`} className="flag-icon" />
        <Typography className="country-name">{`${locationName}, ${fullCountryName}`}</Typography>
        <Typography className="document-type">{formatSubtype(documentType)}</Typography>
        <Typography className="document-date">{date}</Typography>
      </Box>

      <Divider className="divider" />

      {/* Search Results */}
      {searchResults.map((result, index) => {
        // Build a list of components including parents and the hit component
        const components = [...result.parents, {
          tag: result.tag,
          eId: result.eId,
          heading: result.heading,
          subheading: result.subheading,
          crossheading: result.crossheading,
          intro: result.intro,
          num: result.num
        }];

        // Determine which components to render by comparing with previousComponents
        const componentsToRender = components.map((component, idx) => {
          const prevComponent = previousComponents[idx];
          if (
            prevComponent &&
            JSON.stringify(component) === JSON.stringify(prevComponent)
          ) {
            // Component is the same as previous, so we omit rendering the title
            return null;
          } else {
            // Component is different or new, so we render it
            return component;
          }
        });

        // Update previousComponents for the next iteration
        previousComponents = components;

        return (
          <Box key={index} className="search-result">
            <Box className="search-result-content">
              {/* Render each component with indentation */}
              {componentsToRender.map((component, idx) => {
                const indentLevel = idx; // Indent based on hierarchy level
                if (!component) {
                  // Component is omitted, but maintain space for indentation
                  return (
                    <div key={idx} style={{ height: '0px' }}></div>
                  );
                } else {
                  const title = formatComponentTitle(component);
                  return title ? (
                    <Typography
                      key={idx}
                      variant="subtitle1"
                      className="search-result-title"
                      style={{ paddingLeft: `${indentLevel * 20}px` }}
                    >
                      {/* Make the heading a clickable link */}
                      <Link
                        component="button"
                        variant="body2"
                        onClick={() => {
                          // Navigate to the DocumentViewer with docId and eId as parameters
                          navigate(`/document/${docId}?eId=${component.eId}`);
                        }}
                        style={{ textDecoration: 'none', color: 'inherit' }}
                      >
                        {title}
                      </Link>
                    </Typography>
                  ) : null;
                }
              })}

              {/* Render the content text indented based on the last non-null component */}
              <Typography
                variant="body2"
                className="search-result-text"
                style={{ paddingLeft: `${components.length * 20}px` }}
              >
                {result.content}
              </Typography>
            </Box>

            {/* Message Icon Button */}
            <IconButton
              ref={(el) => (buttonRefs.current[index] = el)} // Store button reference
              onClick={() => togglePopup(index)}
              className={`message-icon-button ${activePopupIndex === index ? 'active' : ''}`}
            >
              <MessageIcon />
            </IconButton>

            {/* Popup */}
            {activePopupIndex === index && (
              <Box className="popup" style={{ top: `${popupPosition.top}px` }}>
                <IconButton
                  className="close-button"
                  onClick={() => togglePopup(index)}
                  style={{ position: 'absolute', top: '5px', right: '5px' }}
                >
                  <CloseIcon />
                </IconButton>
                <Typography variant="body1" className="popup-text">
                  The document titled "{documentTitle}" pertains to legal regulations in {locationName}, {fullCountryName}. More details can be added here based on the document context.
                </Typography>
              </Box>
            )}
          </Box>
        );
      })}
    </Box>
  );
};

export default SearchResultCard;