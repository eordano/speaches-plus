function summarizeLoraEvidence(report) {
  if (!report || typeof report !== 'object' || Array.isArray(report)) {
    throw new Error('Expected a report object');
  }
  const lines = [];
  if (report.kind === 'structured-native-repair-evaluation-v1') {
    if (report.tests_modified !== false || !/^[a-f0-9]{64}$/.test(report.test_sha256 || '')
        || !Array.isArray(report.jobs) || report.jobs.length !== 2
        || new Set(report.jobs.map(job => job?.label)).size !== 2) {
      throw new Error('Invalid paired native repair report');
    }
    lines.push(['Evidence type', 'Recorded native tests of base and adapter repairs']);
    lines.push(['Test fixture', {path: report.test_path, sha256: report.test_sha256, modified: false}]);
    const totals = {};
    for (const job of report.jobs) {
      const receipt = job.receipt || {};
      const cases = Object.entries(job.cases || {});
      if (!['base', 'adapter'].includes(job.label) || cases.length === 0
          || !/^[a-f0-9]{40}$/.test(job.job?.head_sha || '')
          || receipt.tested_sha !== job.job.head_sha || receipt.head_sha !== job.job.head_sha
          || receipt.base_sha !== job.job.base_sha || receipt.checkout_clean !== true
          || receipt.check_mode !== 'editmode' || !receipt.finished_at
          || !/^[a-f0-9]{64}$/.test(receipt.test_xml_sha256 || '')
          || !/^[a-f0-9]{64}$/.test(job.source_sha256 || '')
          || ![0, 2].includes(receipt.exit_code)
          || cases.some(([, test]) => !test || !['Passed', 'Failed', 'Skipped', 'Inconclusive'].includes(test.result))) {
        throw new Error('Invalid native repair receipt or test cases');
      }
      const passed = cases.filter(([, test]) => test.result === 'Passed').length;
      const failed = cases.filter(([, test]) => test.result === 'Failed').length;
      if ((receipt.exit_code === 0 && passed !== cases.length)
          || (receipt.exit_code === 2 && failed === 0)) {
        throw new Error('Native repair exit code disagrees with test cases');
      }
      totals[job.label] = {passed, total: cases.length};
      lines.push([`${job.label} tests`, `${passed}/${cases.length} passed; ${failed} failed`]);
      lines.push([`${job.label} failures`, cases.filter(([, test]) => test.result !== 'Passed')
        .map(([name, test]) => ({name, result: test.result, label: test.label, failure: test.failure}))]);
      lines.push([`${job.label} receipt`, {worker: job.worker, tested_sha: receipt.tested_sha,
        source_sha256: job.source_sha256, test_xml_sha256: receipt.test_xml_sha256,
        editor_version: receipt.editor_version, finished_at: receipt.finished_at}]);
    }
    const [a, b] = report.jobs;
    const comparable = a.job.base_sha === b.job.base_sha
      && a.receipt.test_filter === b.receipt.test_filter
      && a.receipt.editor_version === b.receipt.editor_version
      && JSON.stringify(Object.keys(a.cases).sort()) === JSON.stringify(Object.keys(b.cases).sort());
    const delta = totals.adapter.passed - totals.base.passed;
    lines.push(['Comparison', comparable
      ? `Adapter passed ${Math.abs(delta)} ${delta < 0 ? 'fewer' : 'more'} tests than base on the recorded fixture.`
      : 'Not comparable: base, test selection, or editor version differs.']);
    lines.push(['Evidence limits', 'Reported receipts and case summaries; this browser does not fetch or verify the XML hashes. Different workers may affect results.']);
    lines.push(['Training eligibility', 'Individual passing tests do not make a failed repair a positive training example.']);
    lines.push(['Meaning', 'One held-out defect is not a campaign success rate or evidence of general improvement.']);
  } else if (report.kind === 'laguna-api-continuation-smoke-v1') {
    if (report.synthetic !== true || !Array.isArray(report.jobs)
        || report.jobs.length < 1 || report.jobs.length > 20) {
      throw new Error('Invalid API continuation smoke report');
    }
    lines.push(['Evidence type', 'Synthetic API training and adapter continuation']);
    lines.push(['Host at observation', report.host || 'Not recorded']);
    for (const [index, job] of report.jobs.entries()) {
      if (!job || typeof job.job_id !== 'string'
          || !['queued', 'running', 'succeeded', 'failed', 'cancelled'].includes(job.status)
          || !/^[a-f0-9]{64}$/.test(job.training_sha256 || '')
          || !/^[a-f0-9]{64}$/.test(job.adapter_sha256 || '')) {
        throw new Error('Invalid API training job evidence');
      }
      const metrics = job.metrics || {};
      lines.push([`Job ${index + 1}`, {
        id: job.job_id, recorded_status: job.status,
        input_adapter: job.input_adapter || 'No input adapter recorded',
        sft_sha256: job.training_sha256, output_adapter_sha256: job.adapter_sha256,
        finite_gradients: metrics.all_grads_finite === true ? 'Reported finite' : 'Not established',
        hot_loaded_at_observation: typeof metrics.hot_loaded === 'boolean' ? metrics.hot_loaded : 'Not recorded',
      }]);
    }
    lines.push(['Training eligibility', 'Synthetic smoke data is not positive work-order evidence.']);
    lines.push(['Quality gate', 'No held-out work-order evaluation is established by API job success.']);
    lines.push(['Meaning', 'This report records API execution and adapter continuation only. It does not prove better bug fixes, concurrent serving capacity, or live adapter promotion.']);
  } else if (['laguna-long-sequence-capacity-failure-v1', 'laguna-serial-training-capacity-observation-v1'].includes(report.schema)) {
    const failed = report.schema === 'laguna-long-sequence-capacity-failure-v1';
    const peak = failed ? report.peak_host_gpu_memory_mib : report.serial_peak_mib;
    if (!Number.isSafeInteger(report.sequence_length) || report.sequence_length < 2
        || !Number.isSafeInteger(peak) || peak <= 0
        || !Number.isInteger(report.exit_code)
        || (failed ? report.exit_code === 0 : report.exit_code !== 0)) {
      throw new Error('Invalid capacity outcome');
    }
    lines.push(['Evidence type', 'Recorded training capacity experiment']);
    lines.push(['Recorded outcome', failed ? 'Failed; training did not complete' : 'Capacity test passed; task quality unmeasured']);
    lines.push(['Sequence length tested', `${report.sequence_length} tokens`]);
    lines.push(['Host at observation', report.host || 'Not recorded']);
    lines.push(['Peak GPU residency', `${peak} MiB`]);
    lines.push(['Memory scope', 'Sampled whole-GPU residency; a failed allocation may exceed the observed peak.']);
    if (failed) {
      lines.push(['Failure', report.error || 'Not recorded']);
      lines.push(['Last completed phase', report.last_completed_phase || 'Not recorded']);
      lines.push(['Recorded diagnosis', report.diagnosis || 'Not recorded']);
    } else {
      lines.push(['Adapter comparison', report.adapter_identical === true
        ? 'Reported identical; not independently verified by this viewer.'
        : 'Identical export not established; numerical changes require evaluation.']);
      lines.push(['Baseline losses', report.baseline_losses || 'Not recorded']);
      lines.push(['Candidate losses', report.serial_losses || 'Not recorded']);
    }
    lines.push(['Checkpoint and training basis', report.basis || 'Not recorded']);
    lines.push(['Execution settings', report.environment || 'Not recorded']);
    lines.push(['Evidence receipts', {binary_sha256: report.binary_sha256 || 'Not recorded', artifacts: report.artifacts || 'Not recorded'}]);
    lines.push(['Training eligibility', 'Synthetic capacity tests are not positive work-order training evidence.']);
    lines.push(['Meaning', report.limitations || 'This experiment does not establish task quality, concurrent serving capacity, or adapter promotion.']);
  } else if (report.schema === 'laguna-checkpoint-capacity-observation-v1') {
    for (const field of ['sequence_length', 'baseline_peak_mib', 'checkpoint_peak_mib']) {
      if (!Number.isSafeInteger(report[field]) || report[field] <= 0) {
        throw new Error(`Invalid capacity measurement: ${field}`);
      }
    }
    lines.push(['Evidence type', 'Recorded training capacity comparison']);
    lines.push(['Sequence length tested', `${report.sequence_length} tokens`]);
    lines.push(['Host at observation', report.host || 'Not recorded']);
    lines.push(['Baseline peak GPU residency', `${report.baseline_peak_mib} MiB`]);
    lines.push(['Checkpointed peak GPU residency', `${report.checkpoint_peak_mib} MiB`]);
    lines.push(['Recorded memory reduction', `${((1 - report.checkpoint_peak_mib / report.baseline_peak_mib) * 100).toFixed(1)}%`]);
    lines.push(['Memory scope', 'Sampled whole-GPU residency, including allocator reservations; not live tensor bytes.']);
    lines.push(['Adapter comparison', report.baseline_adapter_identical === true
      ? 'Reported byte-identical export to the baseline; not independently verified by this viewer.'
      : 'Byte-identical export not established']);
    lines.push(['Checkpoint and training basis', report.basis || 'Not recorded']);
    lines.push(['Execution settings', report.environment || 'Not recorded']);
    lines.push(['Comparison scope', report.comparison_scope || 'Only the recorded runs; broader coverage not established.']);
    lines.push(['Evidence receipts', {
      binary_sha256: report.binary_sha256 || 'Not recorded',
      log_sha256: report.log_sha256 || 'Not recorded',
      adapter_sha256: report.adapter_sha256 || 'Not recorded',
      baseline_artifacts: report.baseline_artifacts || 'Not recorded',
      candidate_artifacts: report.artifacts || 'Not recorded',
    }]);
    lines.push(['Training eligibility', 'Synthetic capacity inputs are not positive work-order training evidence.']);
    lines.push(['Meaning', 'This comparison covers only the recorded sequence and settings. It does not prove longer examples fit, establish model quality, or measure serving capacity during training.']);
  } else if (report.kind === 'verified-tool-training-window-observation-v1') {
    if (!Number.isInteger(report.native_window) || report.native_window < 2
        || !Array.isArray(report.responses)) throw new Error('Invalid training-window observation');
    lines.push(['Evidence type', 'Recorded training-window sizing']);
    lines.push(['Training window at observation', `${report.native_window} tokens`]);
    lines.push(['Source commit at observation', report.source_commit || 'Not recorded']);
    for (const response of report.responses) {
      if (!Number.isSafeInteger(response.prompt_tokens) || response.prompt_tokens < 0
          || !Number.isSafeInteger(response.completion_tokens) || response.completion_tokens < 1) {
        throw new Error('Invalid recorded token counts');
      }
      const total = response.prompt_tokens + response.completion_tokens;
      lines.push([response.response_id || 'Recorded response',
        `${response.prompt_tokens} prompt + ${response.completion_tokens} completion = ${total} tokens; ${total <= report.native_window ? 'fits' : 'does not fit'} the recorded window`]);
    }
    lines.push(['Training submitted at observation', report.training_submitted === true ? 'Reported as submitted' : report.training_submitted === false ? 'No' : 'Not recorded']);
    lines.push(['Next step', report.next_action || 'Measure intact-example training against available GPU memory.']);
    lines.push(['Meaning', 'Token counts do not prove GPU capacity, training eligibility, or work-order success. This historical window does not establish the currently deployed limit.']);
  } else if (report.kind === 'harness-trial-observation-v1') {
    lines.push(['Evidence type', 'Recorded harness trial']);
    lines.push(['Campaign run', String(report.run_id ?? 'Not recorded')]);
    lines.push(['Retry request', String(report.retry_id ?? 'Not recorded')]);
    lines.push(['Runtime at observation', report.runtime || 'Not recorded']);
    lines.push(['Recorded outcome', report.state_at_observation || 'Not recorded']);
    lines.push(['Requeued executions', report.requeued_execution_ids || []]);
    lines.push(['Repair history boundary', String(report.repair_reset_after_event_id ?? 'Not recorded')]);
    lines.push(['Adapter at observation', report.production_adapter === null ? 'None recorded' : 'Not established']);
    for (const review of Array.isArray(report.draft_reviews) ? report.draft_reviews : []) {
      if (!review || typeof review !== 'object' || Array.isArray(review)) continue;
      lines.push(['Draft review: ' + (review.snapshot_sha || 'Unknown snapshot'), {
        review_method: review.kind || 'Not recorded',
        reported_decision: review.decision || 'Not recorded',
        native_checks: review.native_checks || [],
        finding: review.finding || 'Not recorded',
        required_verification: review.required_verification || 'Not recorded',
        limitation: review.limitation || 'Imported review; not independently verified by this viewer.',
      }]);
    }
    lines.push(['Meaning', 'Starting a trial or verifying its runtime does not establish a completed fix or an improvement in work-order success.']);
    lines.push(['Training eligibility', 'No positive training signal established by trial status']);
    lines.push(['Verification limits', report.limits || []]);
  } else if (report.kind === 'recorded-tool-output-compaction-replay-v1') {
    lines.push(['Evidence type', 'Recorded tool-output compaction replay']);
    lines.push(['Repair claim', String(report.claim_id ?? 'Not recorded')]);
    lines.push(['Complete tool exchanges', String(report.complete_tool_exchanges ?? 'Not recorded')]);
    lines.push(['Duplicate exchanges removed', String(report.removed_duplicate_exchanges ?? 'Not recorded')]);
    lines.push(['Serialized characters before', String(report.serialized_characters_before ?? 'Not recorded')]);
    lines.push(['Serialized characters after', String(report.serialized_characters_after ?? 'Not recorded')]);
    lines.push(['Journal SHA-256', report.journal_sha256 || 'Not recorded']);
    lines.push(['Meaning', 'Reduced serialized tool history does not establish lower token usage, faster execution, or better work-order outcomes.']);
    lines.push(['Verification limits', report.limitation || 'Imported replay, not a live performance result.']);
  } else if (report.schema === 'work-order-tool-adherence-observation-v1') {
    lines.push(['Evidence type', 'Recorded work-order tool adherence']);
    lines.push(['Work order', String(report.work_order_id ?? 'Not recorded')]);
    lines.push(['Execution', String(report.execution_id ?? 'Not recorded')]);
    lines.push(['Recorded state', report.state || 'Unknown']);
    lines.push(['Unknown shell calls', String(report.unknown_shell_responses ?? 'Not recorded')]);
    lines.push(['Terminal calls', String(report.terminal_tool_calls ?? 'Not recorded')]);
    lines.push(['Observation scope', report.observation_scope || 'Not recorded']);
    lines.push(['Failure', report.failure || 'Not recorded']);
    lines.push(['Transcript SHA-256', report.log_sha256 || 'Not recorded']);
    lines.push(['Training eligibility', 'Tool adherence observations alone are not positive training evidence.']);
    lines.push(['Verification limits', 'Imported historical observations, not live status or measured improvement from an alias.']);
  } else if (report.kind === 'draft-paired-regression-observation-v1') {
    const manifest = report.manifest || {};
    const comparison = report.comparison || {};
    lines.push(['Evidence type', 'Recorded paired regression on an incomplete draft']);
    lines.push(['Original production base', manifest.original_base || 'Not recorded']);
    lines.push(['Baseline with identical tests', manifest.baseline_head || 'Not recorded']);
    lines.push(['Draft tested', manifest.candidate_head || 'Not recorded']);
    lines.push(['Test identity', report.fixture?.test_files_sha256 || 'Not recorded']);
    lines.push(['Reported regression comparison', {
      assertion_transition_verified: comparison.assertion_transition_verified,
      expected_regressions: comparison.expected_regressions,
      baseline_assertions_observed: comparison.baseline_assertions_observed,
      before_cases: comparison.before_cases, after_cases: comparison.after_cases,
      before_sha256: comparison.before_sha256, after_sha256: comparison.after_sha256,
      findings: comparison.findings,
    }]);
    lines.push(['Execution receipts', report.receipts || []]);
    lines.push(['Meaning', 'The imported comparison reports a test transition on a draft. It does not establish a registered completed candidate or whole-work-order acceptance.']);
    lines.push(['Training eligibility', 'Not established by this draft report']);
    lines.push(['Verification limits', report.limitation || 'Imported evidence is not independently re-executed by this viewer.']);
  } else if (report.kind === 'draft-source-review-observation-v1') {
    lines.push(['Evidence type', 'AI source review of an unfinished draft']);
    lines.push(['Candidate branch ID', String(report.fix_branch_id ?? 'Not recorded')]);
    lines.push(['Repair claim', String(report.claim_event_id ?? 'Not recorded')]);
    lines.push(['Reviewer', report.reviewer || 'Not recorded']);
    lines.push(['Observation', report.observation || 'Not recorded']);
    lines.push(['Finding', report.finding || 'Not recorded']);
    lines.push(['Reviewed source hashes', report.source_sha256 || {}]);
    lines.push(['Training eligibility', 'No positive training signal established by an AI source review']);
    lines.push(['Verification limits', report.limitation || 'Not an executed regression or completed candidate approval.']);
  } else if (report.kind === 'bounded-recovery-observation-v1') {
    lines.push(['Evidence type', 'Recorded bounded agent recovery']);
    lines.push(['Campaign', report.campaign || 'Not recorded']);
    lines.push(['Candidate branch ID', String(report.fix_branch_id ?? 'Not recorded')]);
    lines.push(['Repair events', {claim: report.claim_event_id, result: report.result_event_id}]);
    lines.push(['Runtime', report.runtime || 'Not recorded']);
    lines.push(['Adapter at observation', report.production_adapter === null ? 'None recorded' : 'Not established']);
    lines.push(['Sessions', {primary: report.primary_session, recovery: report.recovery_session}]);
    lines.push(['Last recovery request', report.recovery_last_request_id || 'Not recorded']);
    lines.push(['Stop reason', report.terminal_reason || 'Not recorded']);
    lines.push(['Draft validation', report.draft_feedback || 'Not recorded']);
    lines.push(['Transcript SHA-256', report.log_sha256 || 'Not recorded']);
    lines.push(['Outcome', report.outcome || 'Not recorded']);
    lines.push(['Training eligibility', 'No positive training signal established by a recovery observation']);
    lines.push(['Verification limits', report.limits || 'Historical observation, not live state or an aggregate success rate.']);
  } else if (report.kind === 'laguna-recorded-loop-adapter-comparison-v1') {
    lines.push(['Evidence type', 'Recorded-context base versus adapter generation']);
    lines.push(['Captured request', report.source_request_id || 'Not recorded']);
    lines.push(['Captured prompt SHA-256', report.prompt_sha256 || 'Not recorded']);
    lines.push(['Evaluated adapter SHA-256', report.adapter_sha256 || 'Not recorded']);
    for (const [index, trial] of (report.cases || []).entries()) {
      lines.push([`Generation ${index + 1}`, {
        variant: trial.variant, requested_model: trial.requested_model,
        returned_model: trial.returned_model, request_id: trial.request_id,
        finish_reason: trial.finish_reason, usage: trial.usage,
        response_sha256: trial.response_sha256, proposed_tools: trial.proposed_tools,
      }]);
    }
    lines.push(['Actions executed', 'No execution evidence established by this generation comparison']);
    lines.push(['Review', report.review?.finding || 'Not reviewed']);
    lines.push(['Meaning', 'An adapter response is not a verified code change. Different output does not establish improvement, and evaluation does not identify the production serving model.']);
    lines.push(['Training eligibility', 'No positive training signal established by this comparison']);
    lines.push(['Verification limits', report.limitation || 'Single-context generation is not a campaign success-rate measurement.']);
  } else if (report.kind === 'laguna-compiler-type-probe-v1') {
    lines.push(['Evidence type', 'Offline compiler-diagnostic prompt probe']);
    lines.push(['Probe model', report.model || 'Not recorded']);
    lines.push(['Adapter used', report.adapter === null ? 'None — base model probe' : 'Not established']);
    lines.push(['Code executed', 'None established by this report']);
    for (const [index, trial] of (report.cases || []).entries()) {
      lines.push([`Diagnostic probe ${index + 1}`, {
        condition: trial.condition, thinking_requested: trial.thinking_requested,
        prompt_sha256: trial.prompt_sha256, response_sha256: trial.response_sha256,
        proposed_expressions: trial.expressions,
        reported_source_match: trial.dimensions_match_declarations,
      }]);
    }
    lines.push(['Meaning', 'Proposed expressions and a reported source match are not compiler results or completed work orders.']);
    lines.push(['Verification limits', report.limitations || 'No independent execution evidence.']);
  } else if (report.kind === 'draft-test-feedback-rollout') {
    const draft = report.latest_draft || report;
    lines.push(['Evidence type', 'Recorded native test feedback for an incomplete draft']);
    lines.push(['Draft snapshot tested', draft.snapshot_sha || 'Not recorded']);
    lines.push(['Reported passing cases', String(draft.passed ?? 'Not recorded')]);
    lines.push(['Reported failing cases', String(draft.failed ?? 'Not recorded')]);
    lines.push(['Feedback SHA-256', draft.feedback_sha256 || 'Not recorded']);
    lines.push(['Deployment at observation', report.deployment_state || 'Not recorded']);
    lines.push(['Agent receipt verified', report.agent_consumption_verified === true ? 'Reported as verified' : 'Not verified']);
    lines.push(['Meaning', 'Stored diagnostics and prepared-runtime checks do not prove delivery to a running agent. These test counts do not establish a completed work order or a baseline-to-candidate regression.']);
    lines.push(['Training eligibility', 'Not established by this draft observation.']);
  } else if (report.kind === 'incomplete-draft-behavioral-check-v1') {
    const manifest = report.manifest || {};
    const native = report.native_result || {};
    const receipt = native.result || {};
    const cases = Object.entries(native.cases || {});
    lines.push(['Evidence type', 'Recorded native tests on an incomplete draft']);
    lines.push(['Registered candidate', manifest.registered_head || 'Not recorded']);
    lines.push(['Draft snapshot tested', receipt.tested_sha || 'Not recorded']);
    lines.push(['Worker', receipt.worker_id || 'Not recorded']);
    lines.push(['Test filter', receipt.test_filter || 'Not recorded']);
    lines.push(['Reported passing cases', String(cases.filter(([, value]) => value.result === 'Passed').length)]);
    lines.push(['Reported failing cases', String(cases.filter(([, value]) => value.result === 'Failed').length)]);
    for (const [name, value] of cases.filter(([, value]) => value.result !== 'Passed')) {
      lines.push([`Test: ${name}`, {result: value.result, failure: value.failure}]);
    }
    lines.push(['NUnit XML SHA-256', receipt.test_xml_sha256 || 'Not recorded']);
    lines.push(['Review decision', report.review?.decision || 'Not recorded']);
    for (const request of report.live_feedback_delivery?.requests || []) {
      lines.push(['Recorded feedback delivery', {
        request_id: request.request_id, prompt_sha256: request.prompt_sha256,
        model: request.model, review_present: request.has_review_3,
        native_failure_present: request.has_native_failure,
        draft_identity_present: request.has_draft_identity,
      }]);
    }
    if (report.live_feedback_delivery) {
      lines.push(['Feedback limits', 'Recorded prompt observations establish reported delivery only, not agent understanding, a completed repair, or live status.']);
    }

    lines.push(['Meaning', 'Draft test results do not validate the registered candidate or establish a baseline-to-candidate regression. Passing compilation does not imply passing tests.']);
    lines.push(['Training eligibility', 'Not established by this draft observation.']);
  } else if (report.kind === 'incomplete-draft-native-feedback-pilot-v1') {
    lines.push(['Evidence type', 'Compiler feedback on an incomplete agent draft']);
    lines.push(['Candidate branch ID', String(report.fix_branch_id ?? 'Not recorded')]);
    lines.push(['Registered candidate', report.registered_head_sha || 'Not recorded']);
    lines.push(['Checked draft', report.snapshot_sha || 'Not recorded']);
    lines.push(['Draft manifest SHA-256', report.manifest_sha256 || 'Not recorded']);
    for (const result of report.observed_results || []) {
      lines.push([`Reported compiler job ${result.job_id}`, {
        worker: result.worker, result: result.state, log_sha256: result.log_digest,
        diagnostic: result.diagnostic || 'Inspect the recorded compiler log',
      }]);
    }
    if (report.continuation) {
      lines.push(['Repair continuation', {
        retry_id: report.continuation.retry_id, state_at_export: report.continuation.state,
        prompt_job_ids: report.continuation.observed_prompt_job_ids,
        operator_interruption: report.continuation.operator_interruption,
        interruption_reason: report.continuation.interruption_reason,
      }]);
    }
    if (report.automatic_feedback) {
      const feedback = report.automatic_feedback;
      lines.push(['Automatic feedback snapshot', feedback.snapshot_sha || 'Not recorded']);
      for (const job of feedback.jobs || []) {
        lines.push([`Automatic compiler job ${job.id}`, {
          worker: job.worker, result: job.state, finished_at: job.finished_at,
          log_sha256: job.log_digest,
        }]);
      }
      lines.push(['Reported next claim start', feedback.next_claim_started_at || 'Not recorded']);
      lines.push(['Reported remaining diagnostics', feedback.reported_diagnostics || 'Not recorded']);
    }
    lines.push(['Training eligibility', 'No positive training signal established']);
    lines.push(['Meaning', 'A snapshot preserves unfinished edits for feedback. Compiler checks and a resumed agent do not establish agent completion, a behaviorally verified fix, or improvement from QLoRA.']);
    lines.push(['Verification limits', 'Imported job results are recorded observations, not live status or independently verified receipts.']);
  } else if (report.source_request_id && report.prompt_sha256 && report.executed_actions === false && Array.isArray(report.cases)) {
    lines.push(['Probe model', report.model || 'Not recorded']);
    lines.push(['Captured request', report.source_request_id]);
    lines.push(['Captured prompt SHA-256', report.prompt_sha256]);
    lines.push(['Actions executed', 'No — offline generation only']);
    for (const [index, item] of report.cases.entries()) {
      lines.push([`Sampling replay ${index + 1}`, {temperature: item.temperature, seed: item.seed,
        first_tool: item.first_tool, finish_reason: item.finish_reason, usage: item.usage,
        output_sha256: item.output_sha256}]);
    }
    const hashes = report.cases.map(item => item.output_sha256);
    const comparable = hashes.length >= 2 && hashes.every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash));
    lines.push(['Reported output comparison', comparable ? (new Set(hashes).size === 1 ? 'Identical recorded hashes' : 'Different recorded hashes') : 'Insufficient evidence']);
    lines.push(['Meaning', 'Proposed actions were not executed. Matching hashes do not establish a successful fix, calibrated decision quality, or a statistically significant sampling result.']);
    lines.push(['Verification limits', report.limitation || 'Imported response hashes are not independently verified by this viewer.']);
  } else if (report.provenance === 'operator:laguna-thinking-cohort' && report.origin_item_id) {
    lines.push(['Original work order', report.origin_item_id]);
    lines.push(['Base model', report.base_model || 'Not recorded']);
    lines.push(['Checkpoint revision', report.checkpoint_revision || 'Not recorded']);
    lines.push(['Adapter used', report.adapter === null ? 'None — base model trial'
      : typeof report.adapter === 'string' && report.adapter ? report.adapter : 'Not recorded']);
    lines.push(['Thinking requested', report.enable_thinking === true ? 'Yes' : report.enable_thinking === false ? 'No' : 'Not recorded']);
    lines.push(['Thinking observed', report.thinking_tokens_observed === true ? 'Reported in output; not independently verified by this viewer' : 'Not established']);
    lines.push(['Previous execution', String(report.prior_execution_id ?? 'Not recorded')]);
    lines.push(['Trial work order', String(report.work_order_id ?? 'Not recorded')]);
    lines.push(['Harness runtime', report.runtime || 'Not recorded']);
    lines.push(['Meaning', 'An inference-setting trial is not QLoRA training. Neither a requested setting nor a launched run establishes better work-order performance.']);
    lines.push(['Verification limits', report.limitation || 'No independently verified outcome supplied.']);
  } else if (Array.isArray(report.branches) && Number.isInteger(report.denominator)) {
    const total = report.denominator;
    const authored = report.autonomous_changesets;
    const compiled = report.autonomous_changesets_with_both_native_compiles_passed;
    if (total <= 0 || !Number.isInteger(authored) || !Number.isInteger(compiled)
        || compiled < 0 || authored < compiled || authored > total) {
      throw new Error('Invalid campaign outcome counts');
    }
    lines.push(['Campaign audit timestamp', report.checked_at || 'Not recorded']);
    lines.push(['Original work orders in scope', String(total)]);
    lines.push(['Independent successes required for 50%', String(Math.ceil(total / 2))]);
    lines.push(['Reported autonomous changesets', `${authored} / ${total} (${(100 * authored / total).toFixed(1)}%)`]);
    lines.push(['Reported changesets compiling on both platforms', `${compiled} / ${total}`]);
    lines.push(['Reported verified fixes', report.verified_fixes == null ? 'Not measured' : String(report.verified_fixes)]);
    lines.push(['Classification basis', report.classification_basis || 'Not recorded']);
    lines.push(['Verification limits', report.verified_fix_explanation || 'This imported audit has not been independently verified by the viewer.']);
    lines.push(['Meaning', 'Changesets and compile passes are intermediate outcomes, not independent successes. Imported counts do not establish that the 50% goal is met.']);
    lines.push(['Next step', 'Reproduce the defect, verify the candidate against the original acceptance criteria, and review the exact commit before counting an independent success.']);
  } else if (Array.isArray(report.origin_item_ids) && Array.isArray(report.excluded)
      && Number.isInteger(report.minimum_independent_groups)) {
    lines.push(['Reported dataset readiness', report.status || 'Not recorded']);
    lines.push(['Eligible code-edit examples', String(report.eligible_rows ?? 'Not recorded')]);
    lines.push(['Independent source work orders', String(report.independent_groups ?? 'Not recorded')]);
    lines.push(['Minimum source work orders for this pilot', String(report.minimum_independent_groups)]);
    lines.push(['Training submitted at observation', report.training_submitted === true ? 'Yes' : report.training_submitted === false ? 'No' : 'Not recorded']);
    lines.push(['Serving adapter at observation', report.production_adapter === null ? 'None recorded; base model only' : typeof report.production_adapter === 'string' && report.production_adapter ? report.production_adapter : 'Not recorded']);
    lines.push(['Original work-order IDs', report.origin_item_ids]);
    lines.push(['Eligible branches', report.eligible_branches || []]);
    for (const entry of report.excluded) {
      lines.push([`Excluded branch ${entry.branch_id ?? 'unknown'}`, entry.reason || 'No reason recorded']);
    }
    lines.push(['Next step', report.status === 'insufficient_independent_work_orders'
      ? 'Collect behavioral regression evidence from more independent work orders before splitting training and held-out data.'
      : 'Revalidate source evidence before exporting training data; readiness alone does not authorize training or promotion.']);
    lines.push(['Meaning', 'These are reported dataset counts, not successful work-order counts. Multiple edits or retries from one work order remain one independent source.']);
  } else if (report.review && report.head_sha && report.eligible_for_positive_training === false) {
    lines.push(['Candidate branch', String(report.branch_id ?? 'Not recorded')]);
    lines.push(['Candidate commit', report.head_sha]);
    lines.push(['Review commit', report.review.tested_sha || 'Not recorded']);
    lines.push(['Review decision', report.review.decision || 'Not recorded']);
    lines.push(['Review provenance', report.review.identity_source || 'Not recorded']);
    lines.push(['Review findings', report.review.rationale || 'Not recorded']);
    lines.push(['Positive training eligibility', 'Excluded by the recorded review']);
    lines.push(['Native regression executed', report.native_regression_executed === true ? 'Reported as executed' : 'Not recorded as executed']);
    lines.push(['Meaning', 'Source review findings are not compiler or regression-test results. A new candidate requires fresh checks and review.']);
  } else if (report.artifact_audit && report.job_id) {
    const audit = report.artifact_audit;
    const metrics = audit.metrics || {};
    lines.push(['Training job', report.job_id]);
    lines.push(['Recorded training status', audit.status || report.status_at_last_check || 'Unknown']);
    lines.push(['Adapter SHA-256', audit.weights_sha256 || 'Not recorded']);
    lines.push(['Saved weights finite', audit.all_saved_weights_finite === true ? 'Verified by artifact audit' : 'Not verified']);
    lines.push(['Held-out loss: base → adapter',
      Number.isFinite(metrics.eval_loss_base) && Number.isFinite(metrics.eval_loss_adapter)
        ? `${metrics.eval_loss_base} → ${metrics.eval_loss_adapter}` : 'Not recorded']);
    const dataset = report.dataset_manifest || {};
    lines.push(['Training signal', dataset.signal || 'Not recorded']);
    lines.push(['Independent source work orders', Number.isInteger(dataset.independent_groups) ? String(dataset.independent_groups) : 'Not recorded']);
    lines.push(['Meaning', 'Training and loss measurements do not establish a working bug fix.']);
  } else if (report.training_job_id && report.review && Array.isArray(report.responses)) {
    lines.push(['Training job', report.training_job_id]);
    lines.push(['Adapter SHA-256', report.adapter_sha256 || 'Not recorded']);
    lines.push(['Original work order', report.origin_item_id || 'Not recorded']);
    lines.push(['Source commit', report.source_base_sha || 'Not recorded']);
    lines.push(['Review method', report.review.kind || 'Not recorded']);
    lines.push(['Code-quality result', report.review.code_quality_result || 'Not recorded']);
    lines.push(['Checks actually run', report.review.checks_run || 'Not recorded']);
    lines.push(['Base output', report.review.base_code || 'Not reviewed']);
    lines.push(['Candidate output', report.review.candidate_code || 'Not reviewed']);
    lines.push(['Response evidence SHA-256', report.responses_sha256 || 'Not recorded']);
  } else if (report.manifest && report.native_runs && report.comparison) {
    const manifest = report.manifest;
    const comparison = report.comparison;
    lines.push(['Candidate branch', String(manifest.branch_id ?? 'Not recorded')]);
    lines.push(['Original base', manifest.original_base || 'Not recorded']);
    lines.push(['Candidate commit', manifest.candidate_head || 'Not recorded']);
    lines.push(['Tests-only baseline commit', manifest.baseline_head || 'Not recorded']);
    lines.push(['Test filter', manifest.filter || 'Not recorded']);
    for (const [name, run] of Object.entries(report.native_runs)) {
      const receipt = run.result || {};
      const cases = Object.entries(run.cases || {});
      const passed = cases.filter(([, test]) => test.result === 'Passed').length;
      lines.push([`Native run: ${name}`, {
        worker: receipt.worker_id || 'Not recorded',
        tested_commit: receipt.tested_sha || 'Not recorded',
        editor: receipt.editor_version || 'Not recorded',
        passed, total: cases.length,
        failures: cases.filter(([, test]) => test.result !== 'Passed').map(([test, outcome]) => ({test, ...outcome})),
        xml_sha256: receipt.test_xml_sha256 || 'Not recorded',
        checkout_clean: receipt.checkout_clean === true,
      }]);
    }
    lines.push(['Recorded assertion transition', comparison.assertion_transition_verified === true
      ? 'Reported as verified; this viewer does not independently validate the source or XML.'
      : 'Not verified']);
    lines.push(['Regression findings', comparison.findings || 'Not recorded']);
    lines.push(['Conclusion', report.conclusion || 'Not recorded']);
    lines.push(['Meaning', 'Candidate passes alone do not prove a bug fix. The baseline must fail at the intended assertion, with identical tests and source-bound evidence.']);
  } else if (report.mode === 'shadow' && report.client_revision && Array.isArray(report.cases)) {
    lines.push(['Decision model', report.model || 'Not recorded']);
    lines.push(['Jev client revision', report.client_revision]);
    lines.push(['Observation corpus SHA-256', report.corpus_sha256 || 'Not recorded']);
    lines.push(['Observation source', report.corpus_kind || 'Not recorded']);
    lines.push(['Execution authority', 'Shadow suggestions only; this viewer grants no execution authority.']);
    const groups = new Map();
    const actions = ['change_strategy', 'report_blocker', 'wait'];
    for (const entry of report.cases) {
      if (!groups.has(entry.case)) groups.set(entry.case, []);
      groups.get(entry.case).push(entry);
    }
    for (const [name, entries] of groups) {
      const orders = entries.map(entry => entry.choice_order);
      const choices = entries.map(entry => entry.response?.answers?.next_action?.choice);
      const complete = entries.length === 6 && orders.every(order => Array.isArray(order)
        && JSON.stringify([...order].sort()) === JSON.stringify(actions))
        && new Set(orders.map(order => JSON.stringify(order))).size === 6
        && entries.every(entry => !entry.error_type) && choices.every(choice => actions.includes(choice));
      lines.push([`Choice-order audit: ${name}`, {
        responses: entries.length,
        coverage: complete ? 'All six orderings returned decisions' : 'Incomplete; stability unknown',
        stability: complete ? (new Set(choices).size === 1 ? 'Stable across orderings' : 'Changes with choice order') : 'Unknown',
        suggestions: [...new Set(choices.filter(choice => actions.includes(choice)))],
        meaning: 'Repeated orderings of one observation are not independent work-order successes.',
      }]);
    }
    for (const entry of report.cases) {
      const answer = entry.response?.answers?.next_action;
      lines.push([`Decision: ${entry.case}`, {
        suggestion: answer?.choice || 'No decision returned',
        choice_order: entry.choice_order || null,
        reported_confidence: answer?.confidence ?? 'Not recorded',
        reported_act_probability: typeof entry.server_action?.act_probability === 'number'
          && Number.isFinite(entry.server_action.act_probability)
          && entry.server_action.act_probability >= 0 && entry.server_action.act_probability <= 1
          ? entry.server_action.act_probability : 'Not recorded or invalid',
        action_signal_meaning: 'Server diagnostic only; not calibrated correctness or permission to execute.',
        probabilities: answer?.probabilities || {},
        expected_action: entry.expected ?? 'No independently verified label supplied',
        error: entry.error_type || null,
        provenance: entry.provenance || 'Not recorded',
      }]);
    }
    lines.push(['Meaning', 'Model confidence is not calibrated accuracy. Suggestions and label matches do not establish successful execution or a working bug fix.']);
  } else {
    throw new Error('Unsupported report. Select a campaign-outcome, dataset-readiness, review-exclusion, training, paired-inference, native regression, or Laya shadow-decision report.');
  }
  lines.push(['Current serving adapter', 'Not queried. An exported report cannot establish which adapter live agents use.']);
  lines.push(['Promotion', report.promotion_allowed === false ? 'Not allowed by this report' : 'No promotion authorization established']);
  lines.push(['Independent work-order success', 'Not established by this report type. A reviewed, behaviorally verified campaign outcome is required.']);
  return lines;
}

if (typeof module !== 'undefined') module.exports = { summarizeLoraEvidence };
if (typeof document !== 'undefined') {
  const input = document.getElementById('evidence-files');
  const output = document.getElementById('evidence-results');
  let generation = 0;
  input.addEventListener('change', async () => {
    const selected = ++generation;
    output.replaceChildren();
    for (const file of Array.from(input.files)) {
      const card = document.createElement('article');
      const title = document.createElement('h3');
      title.textContent = file.name;
      card.append(title);
      try {
        if (file.size > 2 * 1024 * 1024) throw new Error('Report exceeds 2 MiB. Select the summary report, not raw transcripts.');
        const rows = summarizeLoraEvidence(JSON.parse(await file.text()));
        if (generation !== selected) return;
        const list = document.createElement('dl');
        for (const [label, value] of rows) {
          const term = document.createElement('dt');
          term.textContent = label;
          const description = document.createElement('dd');
          description.textContent = typeof value === 'string' ? value : JSON.stringify(value);
          description.style.overflowWrap = 'anywhere';
          list.append(term, description);
        }
        card.append(list);
      } catch (error) {
        const message = document.createElement('p');
        message.textContent = `Cannot read report: ${error.message}`;
        card.append(message);
      }
      if (generation !== selected) return;
      output.append(card);
    }
  });
}
