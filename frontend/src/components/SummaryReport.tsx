'use client';

import React from 'react';
import { Analysis, ClassDistribution, AbnormalSegment } from '@/lib/api';

interface SummaryReportProps {
  analysis: Analysis | null;
}

const CLASS_COLORS: Record<string, string> = {
  N: '#22c55e',
  S: '#f59e0b',
  V: '#ef4444',
  F: '#8b5cf6',
  Q: '#6b7280',
};

const CLASS_NAMES: Record<string, string> = {
  N: 'Normal',
  S: 'Supraventricular',
  V: 'Ventricular',
  F: 'Fusion',
  Q: 'Unknown/Paced',
};

function StatCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  return (
    <div className="bg-gray-50 rounded-lg p-4">
      <div className="text-sm text-gray-500">{label}</div>
      <div className={`text-2xl font-bold ${color || 'text-gray-900'}`}>{value}</div>
    </div>
  );
}

function DistributionBar({ distribution }: { distribution: ClassDistribution[] }) {
  return (
    <div className="space-y-2">
      {distribution.map((item) => (
        <div key={item.class_name} className="flex items-center gap-3">
          <div className="w-16 text-sm font-medium text-gray-700">
            {item.class_name}
          </div>
          <div className="flex-1 bg-gray-200 rounded-full h-4 overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${item.percentage}%`,
                backgroundColor: CLASS_COLORS[item.class_name] || '#6b7280',
              }}
            />
          </div>
          <div className="w-20 text-right text-sm text-gray-600">
            {item.count.toLocaleString()} ({item.percentage.toFixed(1)}%)
          </div>
        </div>
      ))}
    </div>
  );
}

function AbnormalSegments({ segments }: { segments: AbnormalSegment[] }) {
  if (!segments || segments.length === 0) {
    return (
      <div className="text-gray-500 text-sm italic">
        No abnormal segments detected.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {segments.slice(0, 5).map((segment, idx) => (
        <div
          key={idx}
          className="bg-amber-50 border border-amber-200 rounded-lg p-3"
        >
          <div className="flex justify-between items-start">
            <div>
              <span className="font-medium text-amber-800">
                Segment {idx + 1}
              </span>
              <span className="ml-2 text-sm text-amber-600">
                ({segment.dominant_class}: {CLASS_NAMES[segment.dominant_class] || segment.dominant_class})
              </span>
            </div>
            <div className="text-sm text-gray-500">
              {segment.duration_sec.toFixed(1)}s duration
            </div>
          </div>
          <div className="mt-1 text-sm text-gray-600">
            Time: {segment.start_time_sec.toFixed(2)}s - {segment.end_time_sec.toFixed(2)}s
          </div>
          <div className="mt-1 text-sm text-gray-600">
            Beats: {segment.abnormal_beat_indices.length} abnormal beats detected
          </div>
          <div className="mt-1 text-xs text-gray-500">
            Confidence: {(segment.avg_confidence * 100).toFixed(1)}%
          </div>
        </div>
      ))}
      {segments.length > 5 && (
        <div className="text-sm text-gray-500 text-center">
          And {segments.length - 5} more segments...
        </div>
      )}
    </div>
  );
}

export default function SummaryReport({ analysis }: SummaryReportProps) {
  if (!analysis) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Analysis Summary
        </h2>
        <div className="text-gray-500 text-center py-8">
          No analysis results to display.
        </div>
      </div>
    );
  }

  const summary = analysis.summary;

  if (!summary) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h2 className="text-xl font-semibold mb-4 text-gray-800">
          Analysis Summary
        </h2>
        <div className="text-gray-500 text-center py-8">
          Analysis is processing...
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-xl font-semibold text-gray-800">
          Analysis Summary
        </h2>
        <div className="flex items-center gap-2">
          {summary.flagged_for_review ? (
            <span className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-sm font-medium">
              Flagged for Review
            </span>
          ) : (
            <span className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium">
              Normal
            </span>
          )}
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard label="Total Beats" value={summary.total_beats.toLocaleString()} />
        <StatCard label="Normal Beats" value={summary.normal_beats.toLocaleString()} color="text-green-600" />
        <StatCard label="Abnormal Beats" value={summary.abnormal_beats.toLocaleString()} color="text-red-600" />
        <StatCard
          label="Overall Confidence"
          value={`${(summary.overall_confidence * 100).toFixed(1)}%`}
          color="text-blue-600"
        />
      </div>

      {/* Processing Info */}
      {analysis.processing_time_sec && (
        <div className="mb-6 text-sm text-gray-500">
          Processing time: {analysis.processing_time_sec.toFixed(2)}s
        </div>
      )}

      {/* Class Distribution */}
      <div className="mb-6">
        <h3 className="text-lg font-medium text-gray-800 mb-3">
          Class Distribution
        </h3>
        <DistributionBar distribution={summary.class_distribution} />
      </div>

      {/* Abnormal Segments */}
      <div>
        <h3 className="text-lg font-medium text-gray-800 mb-3">
          Abnormal Segments ({summary.abnormal_segments.length})
        </h3>
        <AbnormalSegments segments={summary.abnormal_segments} />
      </div>

      {/* Disclaimer */}
      <div className="mt-6 p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-800">
        <strong>Disclaimer:</strong> This is a screening tool for educational/research
        purposes only. Results should be reviewed by a qualified healthcare professional.
        This system does not provide medical diagnosis.
      </div>
    </div>
  );
}
