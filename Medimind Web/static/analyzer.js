document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('analyzerForm');
    const imageUpload = document.getElementById('imageUpload');
    const preview = document.getElementById('preview');
    const imagePreview = document.getElementById('imagePreview');
    const englishOptions = document.getElementById('englishOptions');
    const arabicOptions = document.getElementById('arabicOptions');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const analyzeText = document.getElementById('analyzeText');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultSection = document.getElementById('resultSection');
    const analysisResult = document.getElementById('analysisResult');
    const saveToHistoryBtn = document.getElementById('saveToHistoryBtn');
    const saveSuccess = document.getElementById('saveSuccess');
    const saveError = document.getElementById('saveError');
    const debugInfo = document.getElementById('debugInfo');

    // 🔁 Language toggle
    document.querySelectorAll('input[name="language"]').forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.value === "🇬🇧 English") {
                englishOptions.style.display = 'block';
                arabicOptions.style.display = 'none';
                document.getElementById('professional').checked = true;
            } else {
                englishOptions.style.display = 'none';
                arabicOptions.style.display = 'block';
                document.getElementById('professionalAr').checked = true;
            }
        });
    });

    // 🖼️ Image preview
    imageUpload.addEventListener('change', function() {
        if (this.files && this.files[0]) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                imagePreview.style.display = 'block';
            };
            reader.readAsDataURL(this.files[0]);
        }
    });

    // 🧠 Analyze image
    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        resultSection.style.display = 'none';
        debugInfo.style.display = 'none';

        if (!imageUpload.files || !imageUpload.files[0]) {
            alert('Please select an image file.');
            return;
        }

        const language = document.querySelector('input[name="language"]:checked')?.value;
        const reportType = document.querySelector('input[name="reportType"]:checked')?.value;

        if (!language || !reportType) {
            alert('Please select both language and report type.');
            return;
        }

        const formData = new FormData();
        formData.append('image', imageUpload.files[0]);
        formData.append('language', language);
        formData.append('report_type', reportType);

        try {
            analyzeText.style.display = 'none';
            loadingSpinner.style.display = 'inline-block';
            analyzeBtn.disabled = true;

            const response = await fetch('/analyze_image', { method: 'POST', body: formData });
            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Analysis failed');

            resultSection.style.display = 'block';
            const parsedContent = marked.parse(data.result);
            analysisResult.querySelector('.card-body').innerHTML = parsedContent;
            analysisResult.querySelector('.card-body').dir = (language === "🇪🇬 العربية") ? 'rtl' : 'ltr';

            window.lastAnalysisResult = data.result;

            saveToHistoryBtn.disabled = false;
            saveToHistoryBtn.style.display = 'inline-block';
            saveToHistoryBtn.innerHTML = '<i class="fas fa-save"></i> Save to Medical History';
            saveSuccess.style.display = 'none';
            saveError.style.display = 'none';

            resultSection.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Analysis error:', error);
            debugInfo.style.display = 'block';
            debugInfo.querySelector('pre').textContent = `Error: ${error.message}`;
            alert('An error occurred during analysis. Please try again.');
        } finally {
            analyzeText.style.display = 'inline-block';
            loadingSpinner.style.display = 'none';
            analyzeBtn.disabled = false;
        }
    });

    // 💾 Save to Medical History
    saveToHistoryBtn.addEventListener('click', async function() {
        const button = this;
        button.disabled = true;
        button.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Saving...';

        try {
            const file = imageUpload.files[0];
            if (!file) throw new Error('No file selected');

            // Convert file to Base64 (remove prefix)
            const reader = new FileReader();
            const base64FileData = await new Promise((resolve, reject) => {
                reader.onload = () => {
                    const result = reader.result.split(',')[1]; // remove prefix like data:image/png;base64,
                    resolve(result);
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });

            const reportType = document.querySelector('input[name="reportType"]:checked')?.value;
            const language = document.querySelector('input[name="language"]:checked')?.value;

            if (!reportType || !language) throw new Error('Missing report type or language');

            const response = await fetch('/save_analysis', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    analysis: window.lastAnalysisResult || '',
                    file_data: base64FileData,
                    reportType: reportType,
                    language: language
                })
            });

            const result = await response.json();

            if (response.ok) {
                saveSuccess.style.display = 'block';
                saveError.style.display = 'none';
                button.style.display = 'none';
            } else {
                throw new Error(result.error || 'Failed to save');
            }
        } catch (error) {
            console.error('Save error:', error);
            saveError.style.display = 'block';
            saveSuccess.style.display = 'none';
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-save"></i> Save to Medical History';
        }
    });
});
