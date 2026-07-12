'use client';

import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { BeatPrediction } from '@/lib/api';

const Plot = dynamic(() => import('react-plotly.js'), { ssr: false });

interface ECGViewerProps {
  predictions: BeatPrediction[];
  currentTime?: number;
  onBeatClick?: (beat: BeatPrediction) => void;
  height?: number;
}

const CLASS_COLORS: Record<string, string> = {
  N: '#22c55e', // Green - Normal
  S: '#f59e0b', // Amber - Supraventricular
  V: '#ef4444', // Red - Ventricular
  F: '#8b5cf6', // Purple - Fusion
  Q: '#6b7280', // Gray - Unknown
};

const CLASS_NAMES: Record<string, string> = {
  N: 'Normal',
  S: 'Supraventricular',
  V: 'Ventricular',
  F: 'Fusion',
  Q: 'Unknown/Paced',
};

export default function ECGViewer({
  predictions,
  currentTime = 0,
  onBeatClick,
  height = 400,
}: ECGViewerProps) {
  const { data, layout, config } = useMemo(() => {
    if (!predictions || predictions.length === 0) {
      return {
        data: [],
        layout: {
          title: 'No data available',
          xaxis: { title: 'Time (seconds)' },
          yaxis: { title: 'Amplitude' },
        },
        config: { displayModeBar: true },
      };
    }

    // Create time array from beat timestamps
    const times = predictions.map((p) => p.timestamp_sec);
    const classes = predictions.map((p) => p.predicted_class);
    const confidences = predictions.map((p) => p.confidence);

    // Create color array based on predictions
    const colors = classes.map((c) => CLASS_COLORS[c] || '#6b7280');

    // Create hover text
    const hoverText = predictions.map(
      (p) =>
        `Beat ${p.beat_index}<br>` +
        `Time: ${p.timestamp_sec.toFixed(2)}s<br>` +
        `Class: ${CLASS_NAMES[p.predicted_class]} (${p.predicted_class})<br>` +
        `Confidence: ${(p.confidence * 100).toFixed(1)}%`
    );

    // Main trace (beat positions colored by class)
    const mainTrace = {
      x: times,
      y: confidences,
      mode: 'markers' as const,
      type: 'scatter' as const,
      marker: {
        color: colors,
        size: 8,
        opacity: 0.8,
      },
      text: hoverText,
      hoverinfo: 'text' as const,
      name: 'Beats',
    };

    // Current time indicator
    const currentTrace = {
      x: [currentTime, currentTime],
      y: [0, 1],
      mode: 'lines' as const,
      type: 'scatter' as const,
      line: {
        color: '#3b82f6',
        width: 2,
        dash: 'dash' as const,
      },
      name: 'Current Time',
    };

    const plotLayout = {
      title: {
        text: 'ECG Beat Predictions',
        font: { size: 16 },
      },
      xaxis: {
        title: 'Time (seconds)',
        gridcolor: '#e5e7eb',
      },
      yaxis: {
        title: 'Confidence',
        range: [0, 1.05],
        gridcolor: '#e5e7eb',
      },
      plot_bgcolor: '#ffffff',
      paper_bgcolor: '#ffffff',
      margin: { l: 50, r: 20, t: 40, b: 50 },
      height,
      showlegend: true,
      legend: {
        x: 1,
        xanchor: 'right' as const,
        y: 1,
        bgcolor: 'rgba(255,255,255,0.8)',
      },
      shapes: [
        // Highlight abnormal regions
        ...predictions
          .filter((p) => p.predicted_class !== 'N')
          .map((p) => ({
            type: 'rect' as const,
            xref: 'x' as const,
            yref: 'paper' as const,
            x0: p.timestamp_sec - 0.1,
            x1: p.timestamp_sec + 0.1,
            y0: 0,
            y1: 1,
            fillcolor: CLASS_COLORS[p.predicted_class] || '#6b7280',
            opacity: 0.1,
            line: { width: 0 },
          })),
      ],
    };

    return {
      data: [mainTrace, currentTrace],
      layout: plotLayout,
      config: { displayModeBar: true, responsive: true },
    };
  }, [predictions, currentTime, height]);

  const handleClick = (event: { points?: Array<{ pointIndex: number }> }) => {
    if (event.points && event.points[0] && onBeatClick) {
      const pointIndex = event.points[0].pointIndex;
      if (pointIndex < predictions.length) {
        onBeatClick(predictions[pointIndex]);
      }
    }
  };

  if (!predictions || predictions.length === 0) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
        No ECG data to display. Run an analysis first.
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-4">
      <Plot
        data={data}
        layout={layout}
        config={config}
        onClick={handleClick}
        style={{ width: '100%' }}
      />
      <div className="mt-2 flex flex-wrap gap-2 text-sm">
        {Object.entries(CLASS_NAMES).map(([code, name]) => (
          <div key={code} className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: CLASS_COLORS[code] }}
            />
            <span>
              {code}: {name}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
