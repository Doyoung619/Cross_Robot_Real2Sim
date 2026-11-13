# Future-action validity correction

The initial development attempt masked invalid current IK poses but omitted invalid lookahead endpoints for 64 round-1 and 44 round-2 new labels (~1%). It was interrupted before final evaluation, archived under history/initial_future_mask_bug, and excluded from final feedback results. The valid baseline was reused. All learned rounds restart from the original100-update adapter using the corrected frozen processor and full current/lookahead masks. No final holdout outcomes were used for the correction.
