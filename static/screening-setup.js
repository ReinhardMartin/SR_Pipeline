function screeningSetupDetails(screening, paperCount = null, criterionCount = 0) {
  const llm = screening.llm;
  const panel = screening.panel;
  const perPaper = screening.backend === 'llm'
    ? (llm.execution_mode === 'per_criterion' ? criterionCount : 1)
    : 0;
  let minimumCalls = paperCount == null ? null : perPaper * paperCount;
  let maximumCalls = minimumCalls;
  const warnings = [];
  const outputs = [];

  if (screening.backend === 'llm') {
    if (llm.include_reason) outputs.push('reason');
    if (llm.extract_evidence) outputs.push('verified quote');
    if (llm.include_confidence_score) outputs.push('self-reported confidence');
    if (llm.execution_mode === 'all_criteria' && llm.allow_uncertainty) {
      warnings.push('Uncertainty is ignored in overall mode, which returns only Include or Exclude.');
    }
    if (llm.execution_mode === 'per_criterion' && !llm.extract_evidence) {
      warnings.push('Evidence quotes are disabled; automatically excluded papers should be checked by a human.');
    }
    if (llm.include_confidence_score) {
      warnings.push('LLM confidence is self-reported and is not a calibrated probability.');
    }
  }

  if (panel.enabled) {
    for (const role of ['second_reviewer', 'judge']) {
      const actor = panel[role];
      if (actor.kind === 'llm' && paperCount != null) {
        maximumCalls += paperCount;
        if (role === 'second_reviewer') minimumCalls += paperCount;
      }
      if (actor.kind === 'llm' && screening.backend === 'llm' && actor.llm?.model === llm.model) {
        warnings.push(`${role.replace('_', ' ')} uses the same model as the primary screener.`);
      }
    }
    if (panel.second_reviewer.kind === 'llm' && panel.judge.kind === 'llm') {
      warnings.push('An all-model panel is experimental; model agreement is not human validation.');
    }
  }

  const primary = screening.backend === 'nli'
    ? 'NLI, one assessment per criterion'
    : `LLM ${llm.model}, ${llm.execution_mode === 'per_criterion' ? 'one call per criterion' : 'one overall call per paper'}`;
  return {
    primary,
    outputs: screening.backend === 'llm' ? (outputs.join(', ') || 'decision only') : 'NLI class scores',
    secondReviewer: panel.enabled ? panel.second_reviewer.kind.toUpperCase() : 'None',
    judge: panel.enabled ? panel.judge.kind.toUpperCase() : 'None',
    minimumCalls,
    maximumCalls,
    warnings,
  };
}

