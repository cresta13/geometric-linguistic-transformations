# GLT-STEER Headline Confidence Intervals

Derived CI audit for the Track 4 headline tables. The script uses Wilson 95% confidence intervals for binary rates, computed from raw rows when available and from summary rate + row counts for strict summary-only metrics.

Script:

- `scripts/summarize_glt_steer_headline_ci.py`

Output:

- `csv/glt_steer_headline_ci.csv`

Key interpretation:

- GPT-2 final-marker logit steering has non-overlapping qualitative separation: no-steering rows remain at `0.0000`, while target steering is high for `?`, `!`, and `...`.
- DistilGPT-2 keeps the no-steering baseline at `0.0000`, but target steering is marker- and layer-sensitive, strongest for `!`, intermediate for `...`, and weakest for `?`.
- The position audit has enough rows (`N=480` per aggregate condition) to show that first/middle/last single prompt-token edits are null while repeated last-token and all-prompt-token edits are strong.
- Composition rows have smaller cells (`N=40` per prompt-style/control row); order contrasts should therefore remain descriptive and should not be promoted as algebraic evidence.

Selected headline rows:

| section | group | metric | estimate | N | 95% CI |
|---|---|---|---:|---:|---|
| `distilgpt2_final_marker_logit_audit` | `target_class=ellipsis \| control=none` | `target_marker_hit` | `0.0000` | `144` | `[0.0000, 0.0260]` |
| `distilgpt2_final_marker_logit_audit` | `target_class=ellipsis \| control=target` | `target_marker_hit` | `0.5139` | `144` | `[0.4330, 0.5941]` |
| `distilgpt2_final_marker_logit_audit` | `target_class=exclamation \| control=none` | `target_marker_hit` | `0.0000` | `144` | `[0.0000, 0.0260]` |
| `distilgpt2_final_marker_logit_audit` | `target_class=exclamation \| control=target` | `target_marker_hit` | `0.7778` | `144` | `[0.7032, 0.8380]` |
| `distilgpt2_final_marker_logit_audit` | `target_class=question \| control=none` | `target_marker_hit` | `0.0000` | `144` | `[0.0000, 0.0260]` |
| `distilgpt2_final_marker_logit_audit` | `target_class=question \| control=target` | `target_marker_hit` | `0.2986` | `144` | `[0.2299, 0.3778]` |
| `gpt2_final_marker_logit_audit` | `target_class=ellipsis \| control=none` | `target_marker_hit` | `0.0000` | `96` | `[0.0000, 0.0385]` |
| `gpt2_final_marker_logit_audit` | `target_class=ellipsis \| control=target` | `target_marker_hit` | `0.8750` | `96` | `[0.7941, 0.9270]` |
| `gpt2_final_marker_logit_audit` | `target_class=exclamation \| control=none` | `target_marker_hit` | `0.0000` | `96` | `[0.0000, 0.0385]` |
| `gpt2_final_marker_logit_audit` | `target_class=exclamation \| control=target` | `target_marker_hit` | `0.9062` | `96` | `[0.8313, 0.9499]` |
| `gpt2_final_marker_logit_audit` | `target_class=question \| control=none` | `target_marker_hit` | `0.0000` | `96` | `[0.0000, 0.0385]` |
| `gpt2_final_marker_logit_audit` | `target_class=question \| control=target` | `target_marker_hit` | `0.8542` | `96` | `[0.7700, 0.9111]` |
| `gpt2_question_exclamation_order_contrast` | `prompt_style=copy_sentence` | `ab_ba_marker_profile_equal` | `0.6500` | `40` | `[0.4951, 0.7787]` |
| `gpt2_question_exclamation_order_contrast` | `prompt_style=copy_sentence` | `ab_equals_ba` | `0.2000` | `40` | `[0.1050, 0.3476]` |
| `gpt2_question_exclamation_order_contrast` | `prompt_style=repeat_sentence` | `ab_ba_marker_profile_equal` | `0.7250` | `40` | `[0.5716, 0.8389]` |
| `gpt2_question_exclamation_order_contrast` | `prompt_style=repeat_sentence` | `ab_equals_ba` | `0.3000` | `40` | `[0.1807, 0.4543]` |
| `gpt2_question_exclamation_order_contrast` | `prompt_style=same_sentence` | `ab_ba_marker_profile_equal` | `0.6500` | `40` | `[0.4951, 0.7787]` |
| `gpt2_question_exclamation_order_contrast` | `prompt_style=same_sentence` | `ab_equals_ba` | `0.2000` | `40` | `[0.1050, 0.3476]` |
| `gpt2_question_position_intervention` | `control=negative_last_each_step \| position_mode=last_each_step` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=negative_last_each_step \| position_mode=last_each_step` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=none \| position_mode=none` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=none \| position_mode=none` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=random_last_each_step \| position_mode=last_each_step` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=random_last_each_step \| position_mode=last_each_step` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=target_last_each_step \| position_mode=last_each_step` | `question_and_preserved` | `0.7917` | `480` | `[0.7531, 0.8256]` |
| `gpt2_question_position_intervention` | `control=target_last_each_step \| position_mode=last_each_step` | `question_mark_hit` | `0.9604` | `480` | `[0.9390, 0.9745]` |
| `gpt2_question_position_intervention` | `control=target_prompt_all_once \| position_mode=prompt_all` | `question_and_preserved` | `0.7896` | `480` | `[0.7509, 0.8237]` |
| `gpt2_question_position_intervention` | `control=target_prompt_all_once \| position_mode=prompt_all` | `question_mark_hit` | `0.8625` | `480` | `[0.8288, 0.8904]` |
| `gpt2_question_position_intervention` | `control=target_prompt_first \| position_mode=prompt_first` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=target_prompt_first \| position_mode=prompt_first` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=target_prompt_last_once \| position_mode=prompt_last` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=target_prompt_last_once \| position_mode=prompt_last` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=target_prompt_middle \| position_mode=prompt_middle` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=target_prompt_middle \| position_mode=prompt_middle` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=wrong_last_each_step \| position_mode=last_each_step` | `question_and_preserved` | `0.0000` | `480` | `[0.0000, 0.0079]` |
| `gpt2_question_position_intervention` | `control=wrong_last_each_step \| position_mode=last_each_step` | `question_mark_hit` | `0.0000` | `480` | `[0.0000, 0.0079]` |
