# NanoResearch User Profile

- `profile_id`: persona-ai4science_journal_conservative
- `archetype_seed`: ai4science_journal

## Summary

- Research direction: NLP
- Method preference: Prefer conservative, scientifically grounded methods with careful controls and restrained claims.
- Resource budget: 1xA100 80GB, 5 days
- Writing tone: highly restrained
- Venue/style preference: Nature/Springer journal
- Template preference: nature_springer
- Figure style: composite scientific figure
- Caption style: self-contained dense

## Feedback Priorities

- Most important feedback: Scientific plausibility, evidence gaps, or overstated generality.
- Unacceptable mistakes: Overclaiming biological/physical conclusions or under-specifying data provenance.

## Recommended Router Defaults

- Planning prompt focus: {'resource_budget': '1xA100 80GB', 'feasibility_bias': 'Prefer reproducible designs with explicit assumptions and controlled compute.'}
- Writing prompt focus: {'tone': 'highly restrained', 'claim_strength': 'conservative', 'venue_style': 'Nature/Springer journal', 'figure_style': 'composite scientific figure', 'caption_style': 'self-contained dense'}
