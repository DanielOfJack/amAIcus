import React from 'react';
import SearchResultCard from './SearchResultCard';
import './SearchResultsContainer.css';

const SearchResultsContainer = ({ documents }) => {
  return (
    <div className="search-results-container">
      {documents.length > 0 ? (
        documents.map((doc, index) => (
          <SearchResultCard
            key={index}
            documentTitle={doc.alias}
            countryName={doc.country}  // Pass country code (like 'za-wc011')
            flagUrl={`https://flagcdn.com/w320/${doc.country.split('-')[0]}.png`}
            documentType={doc.subtype}
            date={doc.expression_date}
            searchResults={doc.search_hits}
            docId={doc.doc_id} // Pass docId to SearchResultCard
          />
        ))
      ) : (
        <p className="no-results">No search results found</p> // Add a class to style
      )}
    </div>
  );
};

export default SearchResultsContainer;