'use client';

import React, { useState, useEffect } from 'react';
import ECGViewer from '@/components/ECGViewer';
import SamplePanel from '@/components/SamplePanel';
import SummaryReport from '@/components/SummaryReport';
import { getAnalysis, Analysis, BeatPrediction, healthCheck } from '@/lib/api';

export default function Home() {
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [predictions, setPredictions] = useState<BeatPrediction[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendUp, setIsBackendUp] = useState<boolean | null>(null);
  const [selectedBeat, setSelectedBeat] = useState<BeatPrediction | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        await healthCheck();
        setIsBackendUp(true);
      } catch {
        setIsBackendUp(false);
      }
    };
    checkBackend();
  }, []);

  // Handle analysis completion
  const handleAnalysisComplete = async (analysisId: string) => {
    setIsLoading(true);
    setError(null);
    
    try {
      // Poll for analysis completion
      let attempts = 0;
      const maxAttempts = 30;
      
      while (attempts < maxAttempts) {
        const result = await getAnalysis(analysisId);
        
        if (result.status === 'completed') {
          setAnalysis(result);
          setPredictions(result.beat_predictions || []);
          setIsLoading(false);
          return;
        }
        
        if (result.status === 'failed') {
          setError('Analysis failed');
          setIsLoading(false);
          return;
        }
        
        // Wait 1 second before next poll
        await new Promise(resolve => setTimeout(resolve, 1000));
        attempts++;
      }
      
      setError('Analysis timed out');
      setIsLoading(false);
    } catch (err: any) {
      console.error('Failed to fetch analysis:', err);
      setError(err.message || 'Failed to fetch analysis results');
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                ECG Arrhythmia Detection
              </h1>
              <p className="text-sm text-gray-500">
                AI-powered heartbeat classification and analysis
              </p>
            </div>
            <div className="flex items-center gap-4">
              {isBackendUp === null && (
                <span className="text-gray-400">Checking backend...</span>
              )}
              {isBackendUp === true && (
                <span className="flex items-center gap-2 text-green-600">
                  <span className="w-2 h-2 bg-green-500 rounded-full"></span>
                  Backend Connected
                </span>
              )}
              {isBackendUp === false && (
                <span className="flex items-center gap-2 text-red-600">
                  <span className="w-2 h-2 bg-red-500 rounded-full"></span>
                  Backend Offline
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        {isBackendUp === false && (
          <div className="mb-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
            <h3 className="text-yellow-800 font-medium">Backend Not Connected</h3>
            <p className="text-yellow-700 text-sm mt-1">
              Please start the FastAPI backend server:
            </p>
            <code className="block mt-2 p-2 bg-yellow-100 rounded text-sm text-yellow-900">
              cd D:\data_science\ecg-project && .venv\Scripts\activate && uvicorn src.api.main:app --host 0.0.0.0 --port 8000
            </code>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Controls */}
          <div className="lg:col-span-1 space-y-6">
            <SamplePanel
              onAnalysisComplete={handleAnalysisComplete}
              isLoading={isLoading}
              setIsLoading={setIsLoading}
            />

            {/* Beat Details */}
            {selectedBeat && (
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-semibold mb-3 text-gray-800">
                  Beat Details
                </h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Beat Index:</span>
                    <span className="font-medium">{selectedBeat.beat_index}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Time:</span>
                    <span className="font-medium">
                      {selectedBeat.timestamp_sec.toFixed(3)}s
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Class:</span>
                    <span className="font-medium">
                      {selectedBeat.predicted_class}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Confidence:</span>
                    <span className="font-medium">
                      {(selectedBeat.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                  {selectedBeat.original_symbol && (
                    <div className="flex justify-between">
                      <span className="text-gray-500">Original:</span>
                      <span className="font-medium">
                        {selectedBeat.original_symbol}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Visualization & Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* ECG Viewer */}
            <ECGViewer
              predictions={predictions}
              onBeatClick={setSelectedBeat}
              height={350}
            />

            {/* Summary Report */}
            <SummaryReport analysis={analysis} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-8">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <p className="text-center text-sm text-gray-500">
            ECG Arrhythmia Detection System - Educational/Research Purpose Only
          </p>
          <p className="text-center text-xs text-gray-400 mt-1">
            Not for clinical use. All results should be reviewed by a qualified healthcare professional.
          </p>
        </div>
      </footer>
    </div>
  );
}
