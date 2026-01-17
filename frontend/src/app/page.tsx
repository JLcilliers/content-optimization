'use client'

import { useState, useCallback } from 'react'
import { Upload, FileText, Sparkles, Download, AlertCircle, CheckCircle2 } from 'lucide-react'
import { analyzeDocument, optimizeAndDownload, type AnalysisResponse, type OptimizationOptions } from '@/lib/api'

type Step = 'upload' | 'analyze' | 'configure' | 'optimize' | 'complete'

export default function Home() {
  const [step, setStep] = useState<Step>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null)
  const [options, setOptions] = useState<OptimizationOptions>({
    primaryKeyword: '',
    secondaryKeywords: [],
    injectKeywords: true,
    generateFaq: true,
    faqCount: 5,
    improveReadability: true,
    optimizeHeadings: true,
    minKeywordDensity: 1.0,
    maxKeywordDensity: 2.5,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const droppedFile = e.dataTransfer.files[0]
    if (droppedFile?.name.endsWith('.docx')) {
      setFile(droppedFile)
      setError(null)
    } else {
      setError('Please upload a .docx file')
    }
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0]
    if (selectedFile?.name.endsWith('.docx')) {
      setFile(selectedFile)
      setError(null)
    } else {
      setError('Please upload a .docx file')
    }
  }, [])

  const handleAnalyze = async () => {
    if (!file) return

    setLoading(true)
    setError(null)

    try {
      const result = await analyzeDocument(
        file,
        options.primaryKeyword || undefined,
        options.secondaryKeywords?.length ? options.secondaryKeywords : undefined
      )
      setAnalysis(result)
      setStep('configure')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }

  const handleOptimize = async () => {
    if (!file || !options.primaryKeyword) return

    setLoading(true)
    setError(null)
    setStep('optimize')

    try {
      const blob = await optimizeAndDownload(file, options)

      // Create download link
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = file.name.replace('.docx', '_optimized.docx')
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)

      setStep('complete')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Optimization failed')
      setStep('configure')
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setStep('upload')
    setFile(null)
    setAnalysis(null)
    setError(null)
  }

  return (
    <main className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      {/* Header */}
      <header className="border-b bg-white/80 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4">
          <h1 className="text-2xl font-bold text-slate-900">
            SEO Content Optimizer
          </h1>
          <p className="text-slate-600">Optimize your content for SEO and AI discoverability</p>
        </div>
      </header>

      <div className="container mx-auto px-4 py-8">
        {/* Progress Steps */}
        <div className="mb-8 flex justify-center">
          <div className="flex items-center space-x-4">
            {['upload', 'analyze', 'configure', 'optimize', 'complete'].map((s, i) => (
              <div key={s} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    step === s
                      ? 'bg-blue-600 text-white'
                      : ['upload', 'analyze', 'configure', 'optimize', 'complete'].indexOf(step) > i
                      ? 'bg-green-600 text-white'
                      : 'bg-slate-200 text-slate-600'
                  }`}
                >
                  {['upload', 'analyze', 'configure', 'optimize', 'complete'].indexOf(step) > i ? (
                    <CheckCircle2 className="w-5 h-5" />
                  ) : (
                    i + 1
                  )}
                </div>
                {i < 4 && (
                  <div
                    className={`w-16 h-1 mx-2 ${
                      ['upload', 'analyze', 'configure', 'optimize', 'complete'].indexOf(step) > i
                        ? 'bg-green-600'
                        : 'bg-slate-200'
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg flex items-center text-red-700">
            <AlertCircle className="w-5 h-5 mr-2" />
            {error}
          </div>
        )}

        {/* Step Content */}
        <div className="max-w-2xl mx-auto">
          {step === 'upload' && (
            <div
              className="border-2 border-dashed border-slate-300 rounded-xl p-12 text-center hover:border-blue-400 transition-colors"
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleFileDrop}
            >
              <Upload className="w-12 h-12 mx-auto text-slate-400 mb-4" />
              <h2 className="text-xl font-semibold text-slate-700 mb-2">
                Upload Your Document
              </h2>
              <p className="text-slate-500 mb-4">
                Drag and drop a .docx file here, or click to select
              </p>
              <input
                type="file"
                accept=".docx"
                onChange={handleFileSelect}
                className="hidden"
                id="file-input"
              />
              <label
                htmlFor="file-input"
                className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700 transition-colors"
              >
                <FileText className="w-5 h-5 mr-2" />
                Select File
              </label>

              {file && (
                <div className="mt-6 p-4 bg-slate-50 rounded-lg">
                  <p className="text-slate-700 font-medium">{file.name}</p>
                  <p className="text-sm text-slate-500">{(file.size / 1024).toFixed(1)} KB</p>
                  <button
                    onClick={() => setStep('analyze')}
                    className="mt-4 px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                  >
                    Continue
                  </button>
                </div>
              )}
            </div>
          )}

          {step === 'analyze' && (
            <div className="bg-white rounded-xl shadow-sm p-8">
              <h2 className="text-xl font-semibold text-slate-700 mb-4">
                Configure Keywords (Optional)
              </h2>
              <p className="text-slate-500 mb-6">
                Enter your target keywords to get more relevant analysis and optimization.
              </p>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Primary Keyword
                  </label>
                  <input
                    type="text"
                    value={options.primaryKeyword}
                    onChange={(e) =>
                      setOptions({ ...options, primaryKeyword: e.target.value })
                    }
                    placeholder="e.g., content optimization"
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">
                    Secondary Keywords (comma-separated)
                  </label>
                  <input
                    type="text"
                    value={options.secondaryKeywords?.join(', ')}
                    onChange={(e) =>
                      setOptions({
                        ...options,
                        secondaryKeywords: e.target.value
                          .split(',')
                          .map((k) => k.trim())
                          .filter(Boolean),
                      })
                    }
                    placeholder="e.g., SEO, rankings, traffic"
                    className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="w-full mt-4 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {loading ? (
                    <span className="flex items-center">
                      <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Analyzing...
                    </span>
                  ) : (
                    'Analyze Document'
                  )}
                </button>
              </div>
            </div>
          )}

          {step === 'configure' && analysis && (
            <div className="space-y-6">
              {/* Score Card */}
              <div className="bg-white rounded-xl shadow-sm p-6">
                <h2 className="text-xl font-semibold text-slate-700 mb-4">Analysis Results</h2>

                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="text-center p-4 bg-slate-50 rounded-lg">
                    <div className="text-3xl font-bold text-blue-600">
                      {analysis.geo_score.toFixed(0)}
                    </div>
                    <div className="text-sm text-slate-600">GEO Score</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 rounded-lg">
                    <div className="text-3xl font-bold text-green-600">
                      {analysis.seo_score.toFixed(0)}
                    </div>
                    <div className="text-sm text-slate-600">SEO Score</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 rounded-lg">
                    <div className="text-3xl font-bold text-purple-600">
                      {analysis.ai_readiness_score.toFixed(0)}
                    </div>
                    <div className="text-sm text-slate-600">AI Readiness</div>
                  </div>
                  <div className="text-center p-4 bg-slate-50 rounded-lg">
                    <div className="text-3xl font-bold text-amber-600">
                      {analysis.readability_score.toFixed(0)}
                    </div>
                    <div className="text-sm text-slate-600">Readability</div>
                  </div>
                </div>

                <div className="text-sm text-slate-600">
                  <p>{analysis.word_count} words | {analysis.paragraph_count} paragraphs | {analysis.heading_count} headings</p>
                </div>
              </div>

              {/* Optimization Options */}
              <div className="bg-white rounded-xl shadow-sm p-6">
                <h2 className="text-xl font-semibold text-slate-700 mb-4">Optimization Options</h2>

                {!options.primaryKeyword && (
                  <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded-lg text-amber-700 text-sm">
                    Please enter a primary keyword to enable optimization
                  </div>
                )}

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-slate-700 mb-1">
                      Primary Keyword *
                    </label>
                    <input
                      type="text"
                      value={options.primaryKeyword}
                      onChange={(e) =>
                        setOptions({ ...options, primaryKeyword: e.target.value })
                      }
                      placeholder="Required for optimization"
                      className="w-full px-4 py-2 border border-slate-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>

                  <div className="grid grid-cols-2 gap-4">
                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={options.injectKeywords}
                        onChange={(e) =>
                          setOptions({ ...options, injectKeywords: e.target.checked })
                        }
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <span className="text-sm text-slate-700">Inject Keywords</span>
                    </label>

                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={options.generateFaq}
                        onChange={(e) =>
                          setOptions({ ...options, generateFaq: e.target.checked })
                        }
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <span className="text-sm text-slate-700">Generate FAQ</span>
                    </label>

                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={options.improveReadability}
                        onChange={(e) =>
                          setOptions({ ...options, improveReadability: e.target.checked })
                        }
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <span className="text-sm text-slate-700">Improve Readability</span>
                    </label>

                    <label className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        checked={options.optimizeHeadings}
                        onChange={(e) =>
                          setOptions({ ...options, optimizeHeadings: e.target.checked })
                        }
                        className="w-4 h-4 text-blue-600 rounded"
                      />
                      <span className="text-sm text-slate-700">Optimize Headings</span>
                    </label>
                  </div>

                  <button
                    onClick={handleOptimize}
                    disabled={!options.primaryKeyword || loading}
                    className="w-full mt-4 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    <Sparkles className="w-5 h-5 mr-2" />
                    Optimize & Download
                  </button>
                </div>
              </div>
            </div>
          )}

          {step === 'optimize' && (
            <div className="bg-white rounded-xl shadow-sm p-12 text-center">
              <svg className="animate-spin mx-auto h-12 w-12 text-blue-600 mb-4" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              <h2 className="text-xl font-semibold text-slate-700">Optimizing Your Content</h2>
              <p className="text-slate-500 mt-2">This may take a moment...</p>
            </div>
          )}

          {step === 'complete' && (
            <div className="bg-white rounded-xl shadow-sm p-12 text-center">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <Download className="w-8 h-8 text-green-600" />
              </div>
              <h2 className="text-xl font-semibold text-slate-700 mb-2">Optimization Complete!</h2>
              <p className="text-slate-500 mb-6">
                Your optimized document has been downloaded. New content is highlighted in green.
              </p>
              <button
                onClick={resetForm}
                className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Optimize Another Document
              </button>
            </div>
          )}
        </div>
      </div>
    </main>
  )
}
