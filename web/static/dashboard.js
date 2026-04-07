(() => {
  const form = document.getElementById('prediction-form');
  if (!form) {
    return;
  }

  const apiUrl = form.dataset.apiUrl;
  const demoValues = {
    N: 90,
    P: 40,
    K: 40,
    pH: 6.5,
    organic_matter: 2.8,
    rainfall: 220,
    temp_min: 18,
    temp_max: 31,
    fertilizer_usage: 120,
    crop_type: 'rice',
  };

  const elements = {
    inputStatus: document.getElementById('input-status'),
    predictionValue: document.getElementById('prediction-value'),
    predictionSubtitle: document.getElementById('prediction-subtitle'),
    riskBadge: document.getElementById('risk-badge'),
    confidenceScore: document.getElementById('confidence-score'),
    confidenceRange: document.getElementById('confidence-range'),
    confidenceFill: document.getElementById('confidence-fill'),
    positiveFactors: document.getElementById('positive-factors'),
    negativeFactors: document.getElementById('negative-factors'),
    explanationSummary: document.getElementById('explanation-summary'),
    narrativeHeadline: document.getElementById('narrative-headline'),
    narrativeHighlights: document.getElementById('narrative-highlights'),
    driverBreakdown: document.getElementById('driver-breakdown'),
    riskReasons: document.getElementById('risk-reasons'),
    stressFactors: document.getElementById('stress-factors'),
    bestCrop: document.getElementById('best-crop'),
    recommendationAlert: document.getElementById('recommendation-alert'),
    selectionNote: document.getElementById('selection-note'),
    cropRanking: document.getElementById('crop-ranking'),
    advisoryList: document.getElementById('advisory-list'),
    demoFill: document.getElementById('demo-fill'),
    demoSubmit: document.getElementById('demo-submit'),
    clearForm: document.getElementById('clear-form'),
    simulatorStatus: document.getElementById('simulator-status'),
    rainfallSlider: document.getElementById('rainfall-slider'),
    fertilizerSlider: document.getElementById('fertilizer-slider'),
    temperatureSlider: document.getElementById('temperature-slider'),
    rainfallSliderValue: document.getElementById('rainfall-slider-value'),
    fertilizerSliderValue: document.getElementById('fertilizer-slider-value'),
    temperatureSliderValue: document.getElementById('temperature-slider-value'),
    baselineYield: document.getElementById('baseline-yield'),
    scenarioYield: document.getElementById('scenario-yield'),
    baselineConfidence: document.getElementById('baseline-confidence'),
    scenarioConfidence: document.getElementById('scenario-confidence'),
    yieldDelta: document.getElementById('yield-delta'),
    riskDelta: document.getElementById('risk-delta'),
    baselineBar: document.getElementById('baseline-bar'),
    scenarioBar: document.getElementById('scenario-bar'),
    scenarioSummary: document.getElementById('scenario-summary'),
    weatherPanel: document.getElementById('weather-panel'),
    weatherLocation: document.getElementById('weather-location'),
    weatherLatitude: document.getElementById('weather-latitude'),
    weatherLongitude: document.getElementById('weather-longitude'),
    fetchWeather: document.getElementById('fetch-weather'),
    weatherStatus: document.getElementById('weather-status'),
  };

  const fieldNames = ['N', 'P', 'K', 'pH', 'organic_matter', 'rainfall', 'temp_min', 'temp_max', 'fertilizer_usage', 'crop_type'];

  function getSelectedMode() {
    const selected = document.querySelector('input[name="input_mode"]:checked');
    return selected ? selected.value : 'manual';
  }

  function updateModeUI() {
    const mode = getSelectedMode();
    if (!elements.weatherPanel) return;

    if (mode === 'weather') {
      elements.weatherPanel.classList.remove('hidden');
      if (elements.weatherStatus) {
        elements.weatherStatus.textContent = 'Weather-assisted mode enabled. Fetch forecast to auto-fill rainfall and temperature.';
      }
    } else {
      elements.weatherPanel.classList.add('hidden');
      if (elements.weatherStatus) {
        elements.weatherStatus.textContent = 'Weather mode inactive.';
      }
    }
  }

  async function fetchWeatherForecast() {
    if (!elements.fetchWeather) return;

    const location = elements.weatherLocation?.value?.trim() || '';
    const latitude = elements.weatherLatitude?.value?.trim() || '';
    const longitude = elements.weatherLongitude?.value?.trim() || '';

    const params = new URLSearchParams();
    if (location) params.set('location', location);
    if (latitude && longitude) {
      params.set('latitude', latitude);
      params.set('longitude', longitude);
    }

    if (!location && !(latitude && longitude)) {
      if (elements.weatherStatus) elements.weatherStatus.textContent = 'Enter location or both latitude/longitude.';
      return;
    }

    elements.fetchWeather.disabled = true;
    if (elements.weatherStatus) elements.weatherStatus.textContent = 'Fetching weather forecast...';

    try {
      const response = await fetch(`/weather-forecast?${params.toString()}`);
      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || 'Weather request failed.');
      }

      if (data.forecastRainfall !== null && data.forecastRainfall !== undefined) {
        form.elements.rainfall.value = Number(data.forecastRainfall).toFixed(2);
        if (elements.rainfallSlider) elements.rainfallSlider.value = Math.round(Number(data.forecastRainfall));
      }
      if (data.forecastTempMin !== null && data.forecastTempMin !== undefined) {
        form.elements.temp_min.value = Number(data.forecastTempMin).toFixed(2);
      }
      if (data.forecastTempMax !== null && data.forecastTempMax !== undefined) {
        form.elements.temp_max.value = Number(data.forecastTempMax).toFixed(2);
      }

      updateSliderLabels();
      await submitPrediction(collectPayload());
      submitSimulation();
      if (elements.weatherStatus) {
        const source = data.source || 'weather service';
        const locationName = data.location?.name || 'selected location';
        if (data.status === 'fallback') {
          elements.weatherStatus.textContent = data.message;
        } else {
          elements.weatherStatus.textContent = `Forecast loaded from ${source} for ${locationName}.`;
        }
      }
    } catch (error) {
      if (elements.weatherStatus) elements.weatherStatus.textContent = error.message;
    } finally {
      elements.fetchWeather.disabled = false;
    }
  }

  function hasCompletePayload() {
    return fieldNames.every((name) => {
      const input = form.elements[name];
      return input && input.value !== '';
    });
  }

  function setStatus(text, tone = 'neutral') {
    if (!elements.inputStatus) return;
    elements.inputStatus.textContent = text;
    elements.inputStatus.dataset.tone = tone;
  }

  function setLoading(isLoading) {
    form.querySelectorAll('input, select, button').forEach((control) => {
      control.disabled = isLoading && control.id !== 'clear-form' && control.id !== 'demo-fill';
    });

    if (isLoading) {
      setStatus('Calculating...', 'loading');
      elements.predictionSubtitle.textContent = 'Generating yield estimate, confidence band, risk, and recommendation...';
    } else {
      setStatus('Ready', 'success');
    }
  }

  function resetList(element, placeholderText) {
    if (!element) return;
    element.innerHTML = `<li class="placeholder">${placeholderText}</li>`;
  }

  function clearOutputs() {
    if (elements.predictionValue) elements.predictionValue.textContent = 'Awaiting prediction';
    if (elements.predictionSubtitle) elements.predictionSubtitle.textContent = 'Enter field values and run the model to see yield, confidence, and recommendations.';
    if (elements.riskBadge) {
      elements.riskBadge.textContent = 'No result';
      elements.riskBadge.className = 'risk-badge neutral';
    }
    if (elements.confidenceScore) elements.confidenceScore.textContent = '--';
    if (elements.confidenceRange) elements.confidenceRange.textContent = '--';
    if (elements.confidenceFill) elements.confidenceFill.style.width = '0%';
    resetList(elements.positiveFactors, 'Awaiting prediction...');
    resetList(elements.negativeFactors, 'Awaiting prediction...');
    resetList(elements.riskReasons, 'No risk analysis yet.');
    resetList(elements.stressFactors, 'No stress factors detected.');
    resetList(elements.advisoryList, 'Actionable advice will appear here after prediction.');
    if (elements.narrativeHeadline) elements.narrativeHeadline.textContent = 'The model will turn the raw prediction into a short decision summary.';
    resetList(elements.narrativeHighlights, 'Narrative insights will appear here after prediction.');
    if (elements.driverBreakdown) elements.driverBreakdown.innerHTML = '<div class="placeholder-card">No signal map yet.</div>';
    if (elements.bestCrop) elements.bestCrop.textContent = '--';
    if (elements.selectionNote) elements.selectionNote.textContent = 'The recommendation engine will compare all supported crops and rank them by expected yield.';
    if (elements.cropRanking) elements.cropRanking.innerHTML = '<div class="placeholder-card">No crop comparison yet.</div>';
    if (elements.explanationSummary) elements.explanationSummary.textContent = 'The model will explain the strongest positive and negative inputs after prediction.';
    setStatus('Waiting for input', 'neutral');
  }

  function fillForm(values) {
    fieldNames.forEach((name) => {
      const input = form.elements[name];
      if (input) input.value = values[name];
    });
  }

  function collectPayload() {
    const payload = {};
    fieldNames.forEach((name) => {
      const input = form.elements[name];
      payload[name] = input.value;
    });
    return payload;
  }

  function collectScenarioPayload() {
    return {
      input: collectPayload(),
      adjustments: {
        rainfall: Number(elements.rainfallSlider?.value || form.elements.rainfall.value),
        fertilizer_usage: Number(elements.fertilizerSlider?.value || form.elements.fertilizer_usage.value),
        temp_delta: Number(elements.temperatureSlider?.value || 0),
      },
    };
  }

  function updateSliderLabels() {
    if (elements.rainfallSlider && elements.rainfallSliderValue) elements.rainfallSliderValue.textContent = `${elements.rainfallSlider.value} mm`;
    if (elements.fertilizerSlider && elements.fertilizerSliderValue) elements.fertilizerSliderValue.textContent = `${elements.fertilizerSlider.value} kg/ha`;
    if (elements.temperatureSlider && elements.temperatureSliderValue) {
      const delta = Number(elements.temperatureSlider.value);
      elements.temperatureSliderValue.textContent = `${delta >= 0 ? '+' : ''}${delta} °C`;
    }
  }

  function renderSimulation(result) {
    if (!result) return;

    const baseline = result.baseline || {};
    const scenario = result.scenario || {};
    const delta = result.delta || {};
    const highest = Math.max(baseline.predictedYield || 0, scenario.predictedYield || 0, 1);

    if (elements.baselineYield) elements.baselineYield.textContent = `${(baseline.predictedYield || 0).toFixed(2)} units`;
    if (elements.scenarioYield) elements.scenarioYield.textContent = `${(scenario.predictedYield || 0).toFixed(2)} units`;
    if (elements.baselineConfidence) {
      elements.baselineConfidence.textContent = `Confidence: ${Math.round((baseline.confidence?.score || 0) * 100)}%`;
    }
    if (elements.scenarioConfidence) {
      elements.scenarioConfidence.textContent = `Confidence: ${Math.round((scenario.confidence?.score || 0) * 100)}%`;
    }
    if (elements.yieldDelta) {
      const change = delta.yield || 0;
      elements.yieldDelta.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} units`;
      elements.yieldDelta.className = `delta-value ${change >= 0 ? 'positive' : 'negative'}`;
    }
    if (elements.riskDelta) {
      const label = delta.riskLabel || 'unchanged';
      elements.riskDelta.textContent = label;
      elements.riskDelta.className = `delta-value ${label === 'improved' ? 'positive' : label === 'worsened' ? 'negative' : 'neutral'}`;
    }
    if (elements.baselineBar) elements.baselineBar.style.width = `${Math.max((baseline.predictedYield || 0) / highest * 100, 8)}%`;
    if (elements.scenarioBar) elements.scenarioBar.style.width = `${Math.max((scenario.predictedYield || 0) / highest * 100, 8)}%`;
    if (elements.scenarioSummary) {
      elements.scenarioSummary.textContent =
        delta.yield >= 0
          ? `The adjusted scenario improves yield by ${delta.yield.toFixed(2)} units.`
          : `The adjusted scenario reduces yield by ${Math.abs(delta.yield).toFixed(2)} units.`;
    }

    if (elements.simulatorStatus) {
      elements.simulatorStatus.textContent = 'Scenario updated';
    }
  }

  let simulationTimer;
  async function submitSimulation() {
    if (!elements.simulatorStatus) return;
    if (!hasCompletePayload()) {
      elements.simulatorStatus.textContent = 'Awaiting inputs';
      return;
    }

    window.clearTimeout(simulationTimer);
    simulationTimer = window.setTimeout(async () => {
      elements.simulatorStatus.textContent = 'Updating scenario...';
      try {
        const response = await fetch('/simulate', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'X-Requested-With': 'fetch',
          },
          body: JSON.stringify(collectScenarioPayload()),
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || 'Simulation request failed.');
        }

        renderSimulation(data);
      } catch (error) {
        elements.simulatorStatus.textContent = 'Scenario error';
        if (elements.scenarioSummary) {
          elements.scenarioSummary.textContent = error.message;
        }
      }
    }, 180);
  }

  function renderFactorList(target, items) {
    if (!target) return;
    target.innerHTML = '';

    if (!items || !items.length) {
      target.innerHTML = '<li class="placeholder">No factor notes yet.</li>';
      return;
    }

    items.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'insight-item';
      li.innerHTML = `
        <div>
          <strong>${item.label || item.feature}</strong>
          <span>${item.message}</span>
        </div>
        <span class="impact ${item.impact >= 0 ? 'positive' : 'negative'}">${item.impact.toFixed(2)}</span>
      `;
      target.appendChild(li);
    });
  }

  function renderNarrative(narrative) {
    if (elements.narrativeHeadline) {
      elements.narrativeHeadline.textContent = narrative?.headline || 'The model will turn the raw prediction into a short decision summary.';
    }

    if (!elements.narrativeHighlights) return;
    elements.narrativeHighlights.innerHTML = '';

    const highlights = narrative?.highlights || [];
    if (!highlights.length) {
      elements.narrativeHighlights.innerHTML = '<li class="placeholder">Narrative insights will appear here after prediction.</li>';
      return;
    }

    highlights.forEach((item) => {
      const li = document.createElement('li');
      li.className = 'narrative-item';
      li.textContent = item;
      elements.narrativeHighlights.appendChild(li);
    });
  }

  function renderDriverBreakdown(drivers) {
    if (!elements.driverBreakdown) return;
    elements.driverBreakdown.innerHTML = '';

    if (!drivers || !drivers.length) {
      elements.driverBreakdown.innerHTML = '<div class="placeholder-card">No signal map yet.</div>';
      return;
    }

    const maxImpact = Math.max(...drivers.map((item) => Math.abs(Number(item.impact) || 0)), 1);

    drivers.slice(0, 6).forEach((item) => {
      const impact = Number(item.impact) || 0;
      const width = Math.max((Math.abs(impact) / maxImpact) * 100, 12);
      const row = document.createElement('div');
      row.className = `driver-row ${impact >= 0 ? 'positive' : 'negative'}`;
      row.innerHTML = `
        <div class="driver-head">
          <div>
            <strong>${item.label}</strong>
            <span>${item.message}</span>
          </div>
          <span class="driver-score">${impact >= 0 ? '+' : ''}${impact.toFixed(2)}</span>
        </div>
        <div class="driver-track" aria-hidden="true">
          <div class="driver-fill" style="width: ${width}%"></div>
        </div>
      `;
      elements.driverBreakdown.appendChild(row);
    });
  }

  function renderRisk(risk) {
    if (!elements.riskBadge) return;
    const tone = risk.level || 'low';
    elements.riskBadge.className = `risk-badge ${tone}`;
    elements.riskBadge.textContent = `${tone.toUpperCase()} RISK`;

    if (elements.riskReasons) {
      elements.riskReasons.innerHTML = '';
      (risk.reasons || []).forEach((reason) => {
        const li = document.createElement('li');
        li.textContent = reason;
        elements.riskReasons.appendChild(li);
      });
      if (!risk.reasons || !risk.reasons.length) {
        elements.riskReasons.innerHTML = '<li class="placeholder">No major risk signals detected.</li>';
      }
    }

    if (elements.stressFactors) {
      elements.stressFactors.innerHTML = '';
      (risk.stressFactors || []).forEach((reason) => {
        const li = document.createElement('li');
        li.textContent = reason;
        elements.stressFactors.appendChild(li);
      });
      if (!risk.stressFactors || !risk.stressFactors.length) {
        elements.stressFactors.innerHTML = '<li class="placeholder">No major stress factors detected.</li>';
      }
    }
  }

  function renderRecommendations(recommendation) {
    if (!recommendation) return;
    if (elements.bestCrop) elements.bestCrop.textContent = recommendation.bestCrop || '--';
    if (elements.selectionNote) elements.selectionNote.textContent = recommendation.selectionNote || recommendation.bestReason || '';

    if (elements.recommendationAlert) {
      if (!recommendation.selectedCrop || !recommendation.bestCrop) {
        elements.recommendationAlert.textContent = 'Awaiting comparison';
        elements.recommendationAlert.dataset.tone = 'neutral';
      } else if (recommendation.selectedCrop === recommendation.bestCrop) {
        elements.recommendationAlert.textContent = 'Selected crop is optimal';
        elements.recommendationAlert.dataset.tone = 'success';
      } else {
        elements.recommendationAlert.textContent = `Better option: ${recommendation.bestCrop}`;
        elements.recommendationAlert.dataset.tone = 'error';
      }
    }

    if (elements.cropRanking) {
      elements.cropRanking.innerHTML = '';
      (recommendation.alternatives || []).forEach((item, index) => {
        const card = document.createElement('div');
        card.className = `ranking-card ${index === 0 ? 'winner' : ''}`;
        card.innerHTML = `
          <div class="ranking-head">
            <strong>${index + 1}. ${item.crop}</strong>
            <span>${item.expectedYield.toFixed(2)} units</span>
          </div>
          <p>${item.reason}</p>
          <small>Suitability score: ${item.suitabilityScore.toFixed(2)}</small>
        `;
        elements.cropRanking.appendChild(card);
      });
    }
  }

  function renderResponse(result) {
    if (!result) return;

    if (elements.predictionValue) elements.predictionValue.textContent = `${result.predictedYield.toFixed(2)} units`;
    if (elements.predictionSubtitle) elements.predictionSubtitle.textContent = result.explanation?.summary || 'Prediction complete.';

    if (elements.confidenceScore) elements.confidenceScore.textContent = `${Math.round((result.confidence?.score || 0) * 100)}%`;
    if (elements.confidenceRange) {
      elements.confidenceRange.textContent = `${result.confidence?.lower?.toFixed(2) ?? '--'} - ${result.confidence?.upper?.toFixed(2) ?? '--'}`;
    }
    if (elements.confidenceFill) elements.confidenceFill.style.width = `${Math.round((result.confidence?.score || 0) * 100)}%`;

    renderRisk(result.risk || {});
    renderFactorList(elements.positiveFactors, result.explanation?.topPositive || []);
    renderFactorList(elements.negativeFactors, result.explanation?.topNegative || []);
    renderNarrative(result.explanation?.narrative || {});
    renderDriverBreakdown(result.explanation?.driverBreakdown || []);
    renderRecommendations(result.recommendation || {});

    if (elements.advisoryList) {
      elements.advisoryList.innerHTML = '';
      (result.advisory || []).forEach((item) => {
        const li = document.createElement('li');
        li.textContent = item;
        elements.advisoryList.appendChild(li);
      });
      if (!result.advisory || !result.advisory.length) {
        elements.advisoryList.innerHTML = '<li class="placeholder">No advisory notes available.</li>';
      }
    }

    if (elements.inputStatus) {
      elements.inputStatus.textContent = 'Prediction complete';
      elements.inputStatus.dataset.tone = 'success';
    }
  }

  async function submitPrediction(payload) {
    setLoading(true);
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'fetch',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Prediction request failed.');
      }

      renderResponse(data);
    } catch (error) {
      if (elements.predictionSubtitle) {
        elements.predictionSubtitle.textContent = error.message;
      }
      if (elements.riskBadge) {
        elements.riskBadge.className = 'risk-badge high';
        elements.riskBadge.textContent = 'ERROR';
      }
      setStatus('Error', 'error');
    } finally {
      setLoading(false);
    }
  }

  form.addEventListener('submit', (event) => {
    event.preventDefault();
    submitPrediction(collectPayload());
  });

  if (elements.demoFill) {
    elements.demoFill.addEventListener('click', () => {
      fillForm(demoValues);
      setStatus('Demo values loaded', 'success');
      updateSliderLabels();
      submitSimulation();
    });
  }

  if (elements.demoSubmit) {
    elements.demoSubmit.addEventListener('click', () => {
      fillForm(demoValues);
      submitPrediction(demoValues);
      updateSliderLabels();
      submitSimulation();
    });
  }

  if (elements.clearForm) {
    elements.clearForm.addEventListener('click', () => {
      form.reset();
      if (elements.rainfallSlider) elements.rainfallSlider.value = '220';
      if (elements.fertilizerSlider) elements.fertilizerSlider.value = '120';
      if (elements.temperatureSlider) elements.temperatureSlider.value = '0';
      clearOutputs();
      updateSliderLabels();
    });
  }

  document.querySelectorAll('input[name="input_mode"]').forEach((radio) => {
    radio.addEventListener('change', updateModeUI);
  });

  if (elements.fetchWeather) {
    elements.fetchWeather.addEventListener('click', fetchWeatherForecast);
  }

  [elements.rainfallSlider, elements.fertilizerSlider, elements.temperatureSlider].forEach((slider) => {
    if (!slider) return;
    slider.addEventListener('input', () => {
      updateSliderLabels();
      submitSimulation();
    });
  });

  form.addEventListener('input', (event) => {
    if (event.target && fieldNames.includes(event.target.name)) {
      submitSimulation();
    }
  });

  clearOutputs();
  updateModeUI();
  updateSliderLabels();

  const initialPredictionElement = document.getElementById('initial-prediction-data');
  if (initialPredictionElement?.dataset?.json) {
    try {
      renderResponse(JSON.parse(initialPredictionElement.dataset.json));
      submitSimulation();
    } catch (error) {
      console.warn('Unable to parse initial prediction payload', error);
    }
  }
})();
