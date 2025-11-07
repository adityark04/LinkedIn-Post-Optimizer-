import { Suggestion } from '../types';

/**
 * Sends a draft post to the custom backend API for optimization.
 * 
 * @param draft The raw text of the LinkedIn post draft.
 * @returns A promise that resolves to an array of optimized suggestions.
 */
export const optimizePostWithDL = async (draft: string): Promise<Suggestion[]> => {
  // For local development, this connects to the Python backend on port 5001.
  const API_ENDPOINT = 'http://127.0.0.1:5001/api/optimize'; 

  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ draft }),
    });

    if (!response.ok) {
      // Catches HTTP errors like 404, 500, etc.
      const errorData = await response.json().catch(() => ({ message: 'An unknown error occurred.' }));
      throw new Error(`Server responded with ${response.status}: ${errorData.message || 'No error message provided.'}`);
    }

    const suggestions: Suggestion[] = await response.json();

    // Basic validation to ensure the response is in the expected format.
    if (!Array.isArray(suggestions)) {
        throw new Error('Invalid response format from the server. Expected an array of suggestions.');
    }

    return suggestions;

  } catch (error) {
    console.error('Error calling the optimization API:', error);
    // Re-throw a more user-friendly error for the UI to catch.
    throw new Error('Could not connect to the optimization server. Please ensure your backend is running and accessible.');
  }
};