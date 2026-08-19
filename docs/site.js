(() => {
  const PARAMETERS = 27_520;
  const TOKENS_PER_STEP = 8 * 96;
  const REPEATED_CORPUS = 25_184;
  const BROAD_CORPUS = 719_847;
  const MAX_BUDGET = 40;

  const budget = document.getElementById("token-budget");
  const budgetOutput = document.getElementById("budget-output");
  const targetTokens = document.getElementById("target-tokens");
  const optimizerSteps = document.getElementById("optimizer-steps");
  const actualTokens = document.getElementById("actual-tokens");
  const repeatedEpochs = document.getElementById("repeated-epochs");
  const broadEpochs = document.getElementById("broad-epochs");
  const repeatedBar = document.getElementById("repeated-bar");
  const broadBar = document.getElementById("broad-bar");
  const formatInteger = new Intl.NumberFormat("en-US");

  function update() {
    const tokensPerParameter = Number(budget.value);
    const target = Math.ceil(PARAMETERS * tokensPerParameter);
    const steps = Math.ceil(target / TOKENS_PER_STEP);
    const actual = steps * TOKENS_PER_STEP;
    const repeated = actual / REPEATED_CORPUS;
    const broad = actual / BROAD_CORPUS;
    const maxRepeatedEpochs = (Math.ceil((PARAMETERS * MAX_BUDGET) / TOKENS_PER_STEP) * TOKENS_PER_STEP) / REPEATED_CORPUS;

    budgetOutput.value = `${tokensPerParameter}×`;
    targetTokens.textContent = formatInteger.format(target);
    optimizerSteps.textContent = formatInteger.format(steps);
    actualTokens.textContent = formatInteger.format(actual);
    repeatedEpochs.textContent = repeated.toFixed(2);
    broadEpochs.textContent = broad.toFixed(2);
    repeatedBar.style.width = `${(repeated / maxRepeatedEpochs) * 100}%`;
    broadBar.style.width = `${Math.max((broad / maxRepeatedEpochs) * 100, 0.8)}%`;
  }

  budget.addEventListener("input", update);
  update();
})();
