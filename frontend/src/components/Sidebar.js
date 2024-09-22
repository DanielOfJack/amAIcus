// src/components/Sidebar.js
import React, { useState } from 'react';
import { Box, Typography, Divider, Button, TextField, Autocomplete, Slider, Chip, List, ListItem, ListItemText, Checkbox } from '@mui/material';
import './Sidebar.css';

// Available labels for "Document" section
const documentLabels = ["By-law", "Act", "Gazette", "Notice", "Judgement", "Ammendment"];

const africanCountries = ["Nigeria", "Kenya", "South Africa", "Ghana", "Egypt", "Ethiopia", "Morocco", "Uganda"];

const cities = ["Lagos", "Nairobi", "Johannesburg", "Accra", "Cairo", "Addis Ababa", "Casablanca", "Kampala"];

const Sidebar = () => {
  const [selectedDocuments, setSelectedDocuments] = useState([]);
  const [selectedCountries, setSelectedCountries] = useState([]);
  const [selectedLocalities, setSelectedLocalities] = useState([]);
  const [yearRange, setYearRange] = useState([1900, 2024]);

  const toggleDocument = (label) => {
    setSelectedDocuments((prev) =>
      prev.includes(label) ? prev.filter((item) => item !== label) : [...prev, label]
    );
  };

  const addCountry = (event, newValue) => {
    if (newValue && !selectedCountries.includes(newValue)) {
      setSelectedCountries([...selectedCountries, newValue]);
    }
  };
  
  const removeCountry = (country) => {
    setSelectedCountries(selectedCountries.filter((c) => c !== country));
  };

  const handleYearRangeChange = (event, newValue) => {
    setYearRange(newValue);
  };

  const toggleLocality = (city) => {
    setSelectedLocalities((prev) =>
      prev.includes(city) ? prev.filter((item) => item !== city) : [...prev, city]
    );
  };

  return (
    <Box className="sidebar-container">
      <Typography variant="h5" className="sidebar-title">Filters</Typography>

      {/* Document Section */}
      <Typography variant="h6" className="sidebar-section-title">Document</Typography>
      <Box className="button-group">
        {documentLabels.map((label) => (
          <Button
            key={label}
            variant={selectedDocuments.includes(label) ? "contained" : "outlined"}
            color={selectedDocuments.includes(label) ? "primary" : "default"}
            onClick={() => toggleDocument(label)}
            className={`custom-button ${selectedDocuments.includes(label) ? "active" : ""}`}
          >
            {label}
          </Button>
        ))}
      </Box>
      <Divider className="sidebar-divider" />

      {/* Jurisdiction Section */}
      <Typography variant="h6" className="sidebar-section-title">Jurisdiction</Typography>
      <Autocomplete
        options={africanCountries}
        onChange={addCountry}
        renderInput={(params) => (
          <TextField 
            {...params} 
            variant="outlined"  /* Ensure label is tied correctly with outlined variant */
            sx={{ backgroundColor: 'white', borderRadius: '20px' }}
          />
        )}
        className="sidebar-autocomplete"
      />
      <Box className="chip-group">
        {selectedCountries.map((country) => (
          <Chip
            key={country}
            label={country}
            onDelete={() => removeCountry(country)}
            className="custom-chip"
          />
        ))}
      </Box>
      <Divider className="sidebar-divider" />

      {/* Locality Section */}
      <Typography variant="h6" className="sidebar-section-title">Locality</Typography>
      <List className="scrollable-list">
        {cities.map((city) => (
          <ListItem key={city} disablePadding>
            <Checkbox
              edge="start"
              checked={selectedLocalities.includes(city)}
              tabIndex={-1}
              onChange={() => toggleLocality(city)}
              className="custom-checkbox"
            />
            <ListItemText primary={city} />
          </ListItem>
        ))}
      </List>
      <Divider className="sidebar-divider" />

      {/* Year Section */}
      <Typography variant="h6" className="sidebar-section-title">Year</Typography>
      <Slider
        value={yearRange}
        onChange={handleYearRangeChange}
        valueLabelDisplay="auto"
        min={1900}
        max={2024}
        className="custom-slider"
      />
      <Typography variant="body2" className="slider-range">
        Range: {yearRange[0]} - {yearRange[1]}
      </Typography>
    </Box>
  );
};

export default Sidebar;