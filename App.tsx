import React, { useState, useCallback, useRef } from 'react';
import { optimizePostWithDL } from './services/apiService';
import { Suggestion } from './types';
import { DRAFT_PLACEHOLDER } from './constants';
import { LightbulbIcon, SparklesIcon, ClipboardIcon } from './components/Icons';
import { Spinner } from './components/Spinner';

const App: React.FC = () => {
  const [draft, setDraft] = useState<string>(DRAFT_PLACEHOLDER);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [suggestions, setSuggestions] = useState<Suggestion[]>([]);
  const [error, setError] = useState<string | null>(null);

  const handleOptimize = useCallback(async () => {
    if (!draft.trim()) {
      setError('Please enter a draft before optimizing.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setSuggestions([]); // Clear previous suggestions
    
    try {
      const result = await optimizePostWithDL(draft);
      setSuggestions(result);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred.';
      setError(errorMessage);
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  }, [draft]);

  return (
    <div className="min-h-screen bg-gray-900 text-gray-200 p-4 sm:p-6 lg:p-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-10">
          <div className="inline-flex items-center justify-center gap-3 mb-4">
            <LightbulbIcon className="w-10 h-10 text-yellow-300" />
            <h1 className="text-4xl sm:text-5xl font-bold tracking-tight bg-gradient-to-r from-yellow-300 to-orange-400 text-transparent bg-clip-text">
              AI LinkedIn Post Optimizer
            </h1>
          </div>
          <p className="text-lg text-gray-400 max-w-2xl mx-auto">
            Transform your raw ideas into engaging, high-impact LinkedIn posts—powered by your custom AI models.
          </p>
        </header>

        <main>
          <div className="bg-gray-800/50 border border-gray-700 rounded-2xl shadow-2xl shadow-gray-950/50 p-6 mb-8">
            <h2 className="text-2xl font-semibold mb-4 text-gray-100">Your Draft Post</h2>
            <textarea
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Start writing your post here..."
              className="w-full h-48 p-4 bg-gray-900/70 border border-gray-600 rounded-lg focus:ring-2 focus:ring-yellow-400 focus:border-yellow-400 transition-colors text-gray-300 resize-none"
              aria-label="Draft post input"
            />
          </div>

          <div className="text-center mb-12">
            <button
              onClick={handleOptimize}
              disabled={isLoading}
              className="relative inline-flex items-center justify-center px-8 py-3 text-lg font-medium text-white bg-gradient-to-r from-yellow-500 to-orange-500 rounded-full shadow-lg hover:shadow-yellow-500/50 hover:scale-105 transition-all duration-300 focus:outline-none focus:ring-4 focus:ring-yellow-500/50 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 group"
            >
              {isLoading ? (
                <>
                  <Spinner className="w-6 h-6 mr-3" />
                  Optimizing...
                </>
              ) : (
                <>
                  <SparklesIcon className="w-6 h-6 mr-3 transition-transform duration-300 group-hover:rotate-12"/>
                  Optimize My Post
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="bg-red-900/50 border border-red-700 text-red-300 px-4 py-3 rounded-lg text-center" role="alert">
              <p>{error}</p>
            </div>
          )}

          {suggestions.length > 0 && (
            <div>
                <h2 className="text-3xl font-bold text-center mb-8 bg-gradient-to-r from-yellow-300 to-orange-400 text-transparent bg-clip-text">Optimized Suggestions</h2>
                <div className="grid grid-cols-1 md:grid-cols-1 gap-6">
                    {suggestions.map((suggestion, index) => (
                        <SuggestionCard key={index} suggestion={suggestion} />
                    ))}
                </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
};

interface SuggestionCardProps {
  suggestion: Suggestion;
}

const SuggestionCard: React.FC<SuggestionCardProps> = ({ suggestion }) => {
    const [copied, setCopied] = useState(false);
    const postContentRef = useRef<HTMLDivElement>(null);

    const handleCopy = useCallback(() => {
        if (postContentRef.current) {
            navigator.clipboard.writeText(postContentRef.current.innerText).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 2000);
            });
        }
    }, []);

    return (
        <div className="bg-gray-800/50 border border-gray-700 rounded-2xl shadow-lg shadow-gray-950/50 overflow-hidden relative">
            <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                    <h3 className="text-xl font-bold text-yellow-300 pr-16">{suggestion.style}</h3>
                     <button
                        onClick={handleCopy}
                        className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1.5 text-sm font-semibold bg-gray-700 hover:bg-yellow-500 hover:text-gray-900 rounded-md transition-all duration-200"
                        aria-label="Copy post text"
                    >
                        <ClipboardIcon className="w-4 h-4" />
                        <span>{copied ? 'Copied!' : 'Copy'}</span>
                    </button>
                </div>
                <div ref={postContentRef} className="whitespace-pre-wrap text-gray-300">
                    {suggestion.post}
                </div>
            </div>
        </div>
    );
};

export default App;