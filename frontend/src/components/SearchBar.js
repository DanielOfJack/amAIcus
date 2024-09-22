import React, { useState } from 'react';
import { Box, TextField, InputAdornment, Select, MenuItem, FormControl, InputLabel, OutlinedInput, Button } from '@mui/material';
import SearchIcon from '@mui/icons-material/Search';
import './SearchBar.css';

const SearchBar = ({ onSearchResults }) => {
  const [searchQuery, setSearchQuery] = useState(""); // State for the search input
  const [sortBy, setSortBy] = useState("Similarity Score");

  const handleSortChange = (event) => {
    setSortBy(event.target.value);
  };

  const handleSearch = async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, location: '', selected_type: sortBy }),
      });

      if (!response.ok) {
        throw new Error("Failed to fetch search results");
      }

      const data = await response.json();
      
      if (data.documents) {
        onSearchResults(data.documents);  // Pass results to parent component (App.js)
      } else {
        console.error("Documents field is missing in the response");
      }

    } catch (error) {
      console.error("Error fetching search results:", error);
    }
  };

  return (
    <Box className="search-bar-container">
      {/* Search Bar */}
      <TextField
        variant="outlined"
        placeholder="Search African Legislation"
        className="search-bar"
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}  // Update search query state
        onKeyPress={(e) => e.key === 'Enter' && handleSearch()}  // Trigger search on Enter key
        InputProps={{
          startAdornment: (
            <InputAdornment position="start">
              <SearchIcon />
            </InputAdornment>
          ),
          sx: {
            backgroundColor: "white",
            color: "black",
            borderRadius: "20px",  // Rounded corners for the search bar
          },
        }}
      />

      {/* Sort By Dropdown */}
      <FormControl variant="outlined" className="sort-by-container">
        <InputLabel shrink>Sort By</InputLabel>
        <Select
          value={sortBy}
          onChange={handleSortChange}
          input={<OutlinedInput label="Sort By" />}
          className="sort-dropdown"
          sx={{
            backgroundColor: "white",
            borderRadius: "20px",  // Rounded corners for the dropdown
          }}
        >
          <MenuItem value="Similarity Score">Similarity Score</MenuItem>
          <MenuItem value="Date (oldest first)">Date (oldest first)</MenuItem>
          <MenuItem value="Date (newest first)">Date (newest first)</MenuItem>
        </Select>
      </FormControl>

      {/* Search Button */}
      <Button
        variant="contained"
        color="primary"
        onClick={handleSearch}  // Trigger search on button click
        sx={{ marginLeft: '10px', borderRadius: '20px' }}
      >
        Search
      </Button>
    </Box>
  );
};

export default SearchBar;