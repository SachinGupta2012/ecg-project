'use client';

import React, { useState } from 'react';
import { analyzeSampleRecord, ModelInfo } from '@/lib/api';

interface SamplePanelProps {
  onAnalysisComplete: (analysisId: string) => void;
  isLoading: boolean;
  setIsLoading: (loading: boolean) => void;
}

const SAMPLE_RECORDS = [
  { id: '100', name: 'Record 100', description: 'Normal sinus rhythm with some APCs' },
  { id: '105', name: 'Record 105', description: 'Contains ventricular ectopic beats' },
  { id: '106', name: 'Record 106', description: 'Ventricular tachycardia episodes' },
  { id: '108', name: 'Record 108', description: 'Bundle branch block' },
  { id: '109', name: 'Record 109', description: 'Ventricular ectopic beats' },
  { id: '111', name: 'Record 111', description: 'Normal with some artifacts' },
  { id: '115', name: 'Record 115', description: 'Normal sinus rhythm' },
  { id: '119', name: 'Record 119', description: 'Ventricular ectopic beats' },
  { id: '200', name: 'Record 200', description: 'Mixed arrhythmias' },
  { id: '207', name: 'Record 207', description: 'Ventricular tachycardia' },
  { id: '217', name: 'Record 217', description: 'Contains PACs and PVCs' },
  { id: '220', name: 'Record 220', description: 'Normal with PVCs' },
  { id: '233', name: 'Record 233', description: 'Ventricular ectopic beats' },
];

export default function SamplePanel({
  onAnalysisComplete,
  isLoading,
  setIsLoading,
}: SamplePanelProps) {
  const [selectedRecord, setSelectedRecord] = useState('100');
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    setIsLoading(true);
    setError(null);

    try {
      const result = await analyzeSampleRecord(selectedRecord);
      onAnalysisComplete(result.analysis_id);
    } catch (err: any) {
      console.error('Analysis failed:', err);
      setError(err.response?.data?.detail || 'Analysis failed. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <h2 className="text-xl font-semibold mb-4 text-gray-800">
        Analyze Sample Record
      </h2>
      
      <p className="text-gray-600 mb-4">
        Select a record from the MIT-BIH Arrhythmia Database to analyze.
      </p>

      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Record
        </label>
        <select
          value={selectedRecord}
          onChange={(e) => setSelectedRecord(e.target.value)}
          className="w-full border border-gray-300 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          disabled={isLoading}
        >
          {SAMPLE_RECORDS.map((record) => (
            <option key={record.id} value={record.id}>
              {record.name} - {record.description}
            </option>
          ))}
        </select>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-md text-red-700 text-sm">
          {error}
        </div>
      )}

      <button
        onClick={handleAnalyze}
        disabled={isLoading}
        className={`w-full py-2 px-4 rounded-md font-medium transition-colors ${
          isLoading
            ? 'bg-gray-400 cursor-not-allowed'
            : 'bg-blue-600 hover:bg-blue-700 text-white'
        }`}
      >
        {isLoading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
                fill="none"
              />
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              />
            </svg>
            Analyzing...
          </span>
        ) : (
          'Run Analysis'
        )}
      </button>

      <div className="mt-4 text-sm text-gray-500">
        <p className="font-medium">About the data:</p>
        <ul className="list-disc list-inside mt-1 space-y-1">
          <li>MIT-BIH Arrhythmia Database (48 half-hour recordings)</li>
          <li>Sampled at 360 Hz with 11-bit resolution</li>
          <li>Beat-by-beat annotations by cardiologists</li>
          <li>~110,000 annotated beats total</li>
        </ul>
      </div>
    </div>
  );
}
