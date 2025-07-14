// SpamGuard Pro JavaScript Application
// Fixed version with proper form handling and AJAX calls

class SpamGuardApp {
    constructor() {
        this.apiBase = '/api';
        this.predictions = [];
        this.stats = {};
        this.metrics = {};
        this.isLoading = false;
        this.toast = null;

        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadInitialData();
        this.startPeriodicUpdates();
        this.initializeToast();
    }

    initializeToast() {
        const toastEl = document.getElementById('toast');
        if (toastEl) {
            this.toast = new bootstrap.Toast(toastEl);
        }
    }

    setupEventListeners() {
        // Prediction form
        const predictionForm = document.getElementById('prediction-form');
        if (predictionForm) {
            predictionForm.addEventListener('submit', (e) => this.handlePrediction(e));
        }

        // Upload form
        const uploadForm = document.getElementById('upload-form');
        if (uploadForm) {
            uploadForm.addEventListener('submit', (e) => this.handleUpload(e));
        }

        // Button handlers
        const clearBtn = document.getElementById('clear-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearForm());
        }

        const exampleBtn = document.getElementById('example-btn');
        if (exampleBtn) {
            exampleBtn.addEventListener('click', () => this.insertExample());
        }

        const refreshBtn = document.getElementById('refresh-history');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', () => this.loadRecentPredictions());
        }

        const healthBtn = document.getElementById('health-check');
        if (healthBtn) {
            healthBtn.addEventListener('click', () => this.performHealthCheck());
        }

        const resetBtn = document.getElementById('reset-model');
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.resetModel());
        }

        const downloadBtn = document.getElementById('download-model');
        if (downloadBtn) {
            downloadBtn.addEventListener('click', () => this.downloadModelInfo());
        }

        // Text input handlers
        const messageInput = document.getElementById('message-input');
        if (messageInput) {
            messageInput.addEventListener('input', () => this.updateTextStats());
        }

        const csvFile = document.getElementById('csv-file');
        if (csvFile) {
            csvFile.addEventListener('change', () => this.validateFile());
        }
    }

    async loadInitialData() {
        try {
            await Promise.all([
                this.loadModelMetrics(),
                this.loadTodayStats(),
                this.loadRecentPredictions()
            ]);
        } catch (error) {
            console.error('Error loading initial data:', error);
        }
    }

    startPeriodicUpdates() {
        // Update stats every 30 seconds
        setInterval(() => {
            this.loadTodayStats();
            this.loadRecentPredictions();
        }, 30000);

        // Update metrics every 5 minutes
        setInterval(() => {
            this.loadModelMetrics();
        }, 300000);
    }

    async handlePrediction(e) {
        e.preventDefault();
        
        const input = document.getElementById('message-input');
        const message = input?.value.trim();

        if (!message || message.length < 3) {
            this.showToast('Please enter a valid message (min 3 characters).', 'warning');
            return;
        }

        this.setLoadingState(true);
        
        try {
            const response = await fetch(`${this.apiBase}/predict`, {
                method: 'POST',
                headers: { 
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                body: JSON.stringify({ message })
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Failed to analyze message');
            }

            const result = await response.json();
            this.displayResults(result);
            this.showToast('Message analyzed successfully!', 'success');
            
            // Refresh stats and history
            this.loadTodayStats();
            this.loadRecentPredictions();
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showToast(`Analysis failed: ${error.message}`, 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    async handleUpload(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('csv-file');
        const file = fileInput?.files[0];

        if (!file) {
            this.showToast('Please select a CSV file.', 'warning');
            return;
        }

        this.setUploadLoadingState(true);
        
        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiBase}/upload-csv`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Upload failed');
            }

            const result = await response.json();
            this.showToast(`Model trained successfully with ${result.samples_processed} samples!`, 'success');
            
            // Clear file input
            fileInput.value = '';
            
            // Refresh metrics
            this.loadModelMetrics();
            
        } catch (error) {
            console.error('Upload error:', error);
            this.showToast(`Upload failed: ${error.message}`, 'error');
        } finally {
            this.setUploadLoadingState(false);
        }
    }

    setLoadingState(loading) {
        const btn = document.getElementById('analyze-btn');
        this.isLoading = loading;
        
        if (btn) {
            btn.disabled = loading;
            btn.innerHTML = loading ?
                '<span class="spinner-border spinner-border-sm me-2"></span>Analyzing...' :
                '<i class="fas fa-search me-1"></i>Analyze Message';
        }
    }

    setUploadLoadingState(loading) {
        const btn = document.getElementById('upload-btn');
        const progress = document.getElementById('upload-progress');
        
        if (btn) {
            btn.disabled = loading;
            btn.innerHTML = loading ?
                '<span class="spinner-border spinner-border-sm me-2"></span>Training...' :
                '<i class="fas fa-cloud-upload-alt me-1"></i>Train Model';
        }
        
        if (progress) {
            progress.style.display = loading ? 'block' : 'none';
        }
    }

    displayResults(result) {
        const resultsCard = document.getElementById('results-card');
        if (!resultsCard) return;

        // Show the results card
        resultsCard.style.display = 'block';

        // Update processing time
        const processingTime = document.getElementById('processing-time');
        if (processingTime) {
            processingTime.textContent = `Processed in ${result.processingTime?.toFixed(2) || '0.00'}s`;
        }

        // Update classification result
        const resultBadge = document.getElementById('result-badge');
        if (resultBadge) {
            const isSpam = result.prediction === 'spam';
            resultBadge.className = `result-badge ${result.prediction}`;
            resultBadge.innerHTML = `
                <i class="fas fa-${isSpam ? 'ban' : 'check-circle'}"></i>
                <span class="ms-2">${result.prediction.toUpperCase()}</span>
            `;
        }

        // Update confidence
        const confidenceValue = document.getElementById('confidence-value');
        const confidenceBar = document.getElementById('confidence-bar');
        if (confidenceValue && confidenceBar) {
            const confidence = Math.round(result.confidence * 100);
            confidenceValue.textContent = `${confidence}%`;
            confidenceValue.className = `confidence-value ${result.prediction}`;
            confidenceBar.style.width = `${confidence}%`;
            confidenceBar.className = `progress-bar ${result.prediction === 'spam' ? 'bg-danger' : 'bg-success'}`;
        }

        // Update feature analysis
        this.updateFeatureAnalysis(result);

        // Update detailed analysis
        this.updateDetailedAnalysis(result);

        // Scroll to results
        resultsCard.scrollIntoView({ behavior: 'smooth' });
    }

    updateFeatureAnalysis(result) {
        const urgencyValue = document.getElementById('urgency-value');
        const promoValue = document.getElementById('promo-value');
        const grammarValue = document.getElementById('grammar-value');
        const sentimentValue = document.getElementById('sentiment-value');

        if (urgencyValue) {
            urgencyValue.textContent = result.urgency_indicators || 'Low';
            urgencyValue.className = `feature-value ${(result.urgency_indicators || 'Low').toLowerCase()}`;
        }

        if (promoValue) {
            promoValue.textContent = result.promotional_keywords || 'Low';
            promoValue.className = `feature-value ${(result.promotional_keywords || 'Low').toLowerCase()}`;
        }

        if (grammarValue) {
            grammarValue.textContent = result.grammar_quality || 'Good';
            grammarValue.className = `feature-value ${(result.grammar_quality || 'Good').toLowerCase()}`;
        }

        if (sentimentValue) {
            const sentiment = result.features?.sentiment || 0;
            let sentimentText = 'Neutral';
            if (sentiment > 0.1) sentimentText = 'Positive';
            else if (sentiment < -0.1) sentimentText = 'Negative';
            
            sentimentValue.textContent = sentimentText;
            sentimentValue.className = `feature-value ${sentimentText.toLowerCase()}`;
        }
    }

    updateDetailedAnalysis(result) {
        const analysis = result.analysis || {};
        const features = result.features || {};

        const lengthValue = document.getElementById('length-value');
        if (lengthValue) {
            lengthValue.textContent = `${analysis.length || 0} chars`;
        }

        const wordsValue = document.getElementById('words-value');
        if (wordsValue) {
            wordsValue.textContent = `${analysis.word_count || 0} words`;
        }

        const sentimentScore = document.getElementById('sentiment-score');
        if (sentimentScore) {
            const sentiment = analysis.sentiment || 0;
            sentimentScore.textContent = sentiment.toFixed(2);
        }

        const uppercaseValue = document.getElementById('uppercase-value');
        if (uppercaseValue) {
            const ratio = (analysis.uppercase_ratio || 0) * 100;
            uppercaseValue.textContent = `${ratio.toFixed(1)}%`;
        }

        const punctuationValue = document.getElementById('punctuation-value');
        if (punctuationValue) {
            punctuationValue.textContent = analysis.punctuation_score || 0;
        }

        const spamKeywordsValue = document.getElementById('spam-keywords-value');
        if (spamKeywordsValue) {
            spamKeywordsValue.textContent = features.spam_keywords || 0;
        }
    }

    clearForm() {
        const input = document.getElementById('message-input');
        if (input) {
            input.value = '';
            this.updateTextStats();
        }
        
        const card = document.getElementById('results-card');
        if (card) {
            card.style.display = 'none';
        }
    }

    insertExample() {
        const examples = [
            "Congratulations! You've won a $1000 gift card! Click here to claim now!",
            "Hi, how are you doing today? Want to grab lunch sometime?",
            "URGENT: Your account will be suspended unless you verify immediately!",
            "Thanks for your help with the project. The meeting went well.",
            "Limited time offer! Get 50% off everything. Act fast!",
            "Can you please send me the report by tomorrow morning?",
            "FREE MONEY! Win big prizes! Call now! Don't miss out!",
            "Meeting has been rescheduled to 3 PM in conference room B."
        ];
        
        const input = document.getElementById('message-input');
        if (input) {
            const randomExample = examples[Math.floor(Math.random() * examples.length)];
            input.value = randomExample;
            this.updateTextStats();
        }
    }

    updateTextStats() {
        const input = document.getElementById('message-input');
        const message = input?.value || '';
        const charCount = message.length;
        const wordCount = message.trim() ? message.trim().split(/\s+/).length : 0;

        const charCountEl = document.getElementById('char-count');
        const wordCountEl = document.getElementById('word-count');

        if (charCountEl) {
            charCountEl.textContent = charCount;
            const color = charCount > 8000 ? '#dc3545' : charCount > 5000 ? '#ffc107' : '#6c757d';
            charCountEl.style.color = color;
        }

        if (wordCountEl) {
            wordCountEl.textContent = wordCount;
        }
    }

    validateFile() {
        const fileInput = document.getElementById('csv-file');
        const file = fileInput?.files[0];
        
        if (!file) return;

        const validTypes = ['text/csv', 'text/plain'];
        const validExtensions = ['.csv', '.txt'];
        
        const hasValidType = validTypes.includes(file.type);
        const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));

        if (!hasValidType && !hasValidExtension) {
            this.showToast('Invalid file format. Please upload a CSV or TXT file.', 'warning');
            fileInput.value = '';
            return;
        }

        if (file.size > 16 * 1024 * 1024) {
            this.showToast('File too large. Maximum size is 16MB.', 'warning');
            fileInput.value = '';
            return;
        }

        this.showToast('File validated successfully.', 'success');
    }

    async loadModelMetrics() {
        try {
            const response = await fetch(`${this.apiBase}/model/metrics`);
            if (!response.ok) return;
            
            const metrics = await response.json();
            this.metrics = metrics;
            this.updateMetricsDisplay(metrics);
        } catch (error) {
            console.error('Error loading metrics:', error);
        }
    }

    updateMetricsDisplay(metrics) {
        const accuracyValue = document.getElementById('accuracy-value');
        const accuracyBar = document.getElementById('accuracy-bar');
        const precisionValue = document.getElementById('precision-value');
        const precisionBar = document.getElementById('precision-bar');
        const recallValue = document.getElementById('recall-value');
        const recallBar = document.getElementById('recall-bar');
        const f1Value = document.getElementById('f1-value');
        const f1Bar = document.getElementById('f1-bar');

        if (accuracyValue && accuracyBar) {
            const accuracy = Math.round(metrics.accuracy * 100);
            accuracyValue.textContent = `${accuracy}%`;
            accuracyBar.style.width = `${accuracy}%`;
        }

        if (precisionValue && precisionBar) {
            const precision = Math.round(metrics.precision * 100);
            precisionValue.textContent = `${precision}%`;
            precisionBar.style.width = `${precision}%`;
        }

        if (recallValue && recallBar) {
            const recall = Math.round(metrics.recall * 100);
            recallValue.textContent = `${recall}%`;
            recallBar.style.width = `${recall}%`;
        }

        if (f1Value && f1Bar) {
            const f1 = Math.round(metrics.f1Score * 100);
            f1Value.textContent = `${f1}%`;
            f1Bar.style.width = `${f1}%`;
        }
    }

    async loadTodayStats() {
        try {
            const response = await fetch(`${this.apiBase}/stats/today`);
            if (!response.ok) return;
            
            const stats = await response.json();
            this.stats = stats;
            this.updateStatsDisplay(stats);
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    updateStatsDisplay(stats) {
        const totalMessages = document.getElementById('total-messages');
        const spamCount = document.getElementById('spam-count');
        const hamCount = document.getElementById('ham-count');
        const avgResponse = document.getElementById('avg-response');

        if (totalMessages) {
            totalMessages.textContent = stats.totalMessages || 0;
        }

        if (spamCount) {
            spamCount.textContent = stats.spamCount || 0;
        }

        if (hamCount) {
            hamCount.textContent = stats.hamCount || 0;
        }

        if (avgResponse) {
            avgResponse.textContent = `${(stats.avgResponseTime || 0).toFixed(2)}s`;
        }
    }

    async loadRecentPredictions() {
        try {
            const response = await fetch(`${this.apiBase}/predictions/recent?limit=10`);
            if (!response.ok) return;
            
            const predictions = await response.json();
            this.predictions = predictions;
            this.updatePredictionsDisplay(predictions);
        } catch (error) {
            console.error('Error loading predictions:', error);
        }
    }

    updatePredictionsDisplay(predictions) {
        const container = document.getElementById('recent-predictions');
        if (!container) return;

        if (predictions.length === 0) {
            container.innerHTML = `
                <div class="text-center text-muted py-3">
                    <i class="fas fa-clock me-2"></i>
                    No recent predictions
                </div>
            `;
            return;
        }

        container.innerHTML = predictions.map(pred => `
            <div class="history-item">
                <div class="history-dot ${pred.prediction}"></div>
                <div class="history-content">
                    <div class="history-message">${pred.message}</div>
                    <div class="history-meta">
                        ${new Date(pred.createdAt).toLocaleString()} • 
                        ${(pred.processingTime * 1000).toFixed(0)}ms
                    </div>
                </div>
                <div class="history-result">
                    <div class="history-prediction ${pred.prediction}">
                        ${pred.prediction.toUpperCase()}
                    </div>
                    <div class="history-confidence">
                        ${Math.round(pred.confidence * 100)}%
                    </div>
                </div>
            </div>
        `).join('');
    }

    showToast(message, type = 'info') {
        const toastEl = document.getElementById('toast');
        const toastBody = toastEl?.querySelector('.toast-body');
        const icon = toastEl?.querySelector('.toast-header i');

        if (!toastEl || !toastBody || !icon) return;

        toastBody.textContent = message;

        const typeIcons = {
            success: 'fas fa-check-circle text-success',
            warning: 'fas fa-exclamation-triangle text-warning',
            error: 'fas fa-times-circle text-danger',
            info: 'fas fa-info-circle text-info'
        };

        icon.className = `${typeIcons[type] || typeIcons.info} me-2`;
        
        if (this.toast) {
            this.toast.show();
        }
    }

    async performHealthCheck() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const health = await response.json();

            if (health.status === 'healthy') {
                this.showToast('System is healthy and running normally.', 'success');
            } else {
                this.showToast('System health check failed.', 'error');
            }
        } catch (error) {
            console.error('Health check error:', error);
            this.showToast('Health check failed. Please try again.', 'error');
        }
    }

    async resetModel() {
        if (!confirm('Are you sure you want to reset the model to default? This cannot be undone.')) {
            return;
        }

        try {
            const response = await fetch(`${this.apiBase}/model/reset`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Reset failed');
            }

            const result = await response.json();
            this.showToast('Model reset successfully.', 'success');
            this.loadModelMetrics();
        } catch (error) {
            console.error('Reset error:', error);
            this.showToast(`Reset failed: ${error.message}`, 'error');
        }
    }

    downloadModelInfo() {
        const info = {
            metrics: this.metrics,
            stats: this.stats,
            timestamp: new Date().toISOString(),
            recentPredictions: this.predictions.slice(0, 5)
        };

        const blob = new Blob([JSON.stringify(info, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `spamguard-model-info-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        this.showToast('Model information downloaded successfully.', 'success');
    }
}

// Initialize the app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new SpamGuardApp();
});
