# CarWise Retrieval and Generation Evaluation

Evaluated: 2026-07-28T09:53:53.308141+00:00

## Summary

- Queries: 9
- Retrieval passes: 9/9
- Generation grounding passes: 9/9
- Manual answer-quality passes: 4/9
- Retrieval pass rate: 100.0%
- Generation grounding pass rate: 100.0%
- Manual answer-quality pass rate: 44.4%
- Generation provider: Local LLM (Qwen/Qwen2.5-0.5B-Instruct)
- Mean retrieval latency: 0.01s
- Mean generation latency: 4.79s

A retrieval pass means every expected evidence term appeared in the top-k chunks. A generation grounding pass means the answer was non-empty and every citation referred to one of the supplied chunks. The qualitative notes below also record whether each result answered the intended constraint.

## Per-query results

### q1: Which used SUVs in Phnom Penh are listed below $20,000?

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: PASS
- Top evidence: 2015 Land Rover Discovery Sport Luxury SUV (0.573), 2012 Not identified RangeEvoque Suv (0.581), 2019 Honda HR-V (0.518)
- Expected evidence found: Asking price, Used, SUV, Phnom Penh
- Performance: retrieval 0.02s; generation 5.37s
- Qualitative assessment: The answer selects a matching used SUV under the budget and keeps its price, location, body type, and engine details within the retrieved evidence.

**Generated answer**

The 2015 Land Rover Discovery Sport Luxury SUV, priced at $18,000, is listed as used and has an economy of 22 mpg combined (EPA). It is located in Tuol Kouk, Phnom Penh, and is equipped with a 4-cylinder engine and a displacement of 2 liters. The vehicle's condition is described as "Used," and it is identified as a SUV.

Grounding references used by the local LLM: [S1] [S2] [S3]

### q2: Recommend a Toyota Prius under $18,000.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: NEEDS_REVIEW
- Top evidence: 2008 Toyota Prius (0.550), 2012 Toyota Prius (0.546), 2012 Toyota Prius (0.544)
- Expected evidence found: Toyota, Prius, Hybrid
- Performance: retrieval 0.01s; generation 3.89s
- Qualitative assessment: The recommendation and price are supported, but the phrase "good condition" is stronger than the seller's generic Used condition label.

**Generated answer**

Based on the provided information, the most relevant recommendation for a Toyota Prius under $18,000 is the 2012 Toyota Prius. It has a price of $15,600, is in good condition, uses a hybrid engine, has an economy of 48 mpg combined, and features an automatic transmission with variable gear ratios.

Grounding references used by the local LLM: [S1] [S2] [S3]

### q3: Find an electric car listed in Cambodia under $30,000.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: NEEDS_REVIEW
- Top evidence: 2026 JAC EV (0.686), 2026 JAC EV (0.682), 2023 Not identified Ev (0.681)
- Expected evidence found: Electricity, Asking price
- Performance: retrieval 0.01s; generation 4.49s
- Qualitative assessment: The vehicle, price, EV powertrain, and location are supported, but "used for range testing" is not present in the evidence.

**Generated answer**

The 2026 JAC EV in Saensokh, Phnom Penh, Cambodia is priced at $29,999 and has a single-speed electric drive, located in a sedan body. It is in good condition and can be used for range testing. The most relevant details include the fuel type (electric), displacement (N/A), and seating capacity (usually 4-5).

Grounding references used by the local LLM: [S1] [S2] [S3]

### q4: Show Kia hatchbacks under $10,000.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: FAIL
- Top evidence: 2005 Kia Morning (0.479), 2013 Kia Starex (0.454), 2012 Kia Morning (0.469)
- Expected evidence found: Kia, Hatchback
- Performance: retrieval 0.01s; generation 3.81s
- Qualitative assessment: The answer focuses on a Kia Starex instead of the stronger Kia Morning hatchback matches. This exposes a body-type normalization error that should be corrected in a future dataset revision.

**Generated answer**

[SA] The 2013 Kia Starex is priced at $8,800 and has a 5-speed manual transmission, making it an economical choice for those on a tight budget.

Grounding references used by the local LLM: [S1] [S2] [S3]

### q5: Which pickup trucks are listed below $15,000?

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: PASS
- Top evidence: 2014 Not identified Karen do WD Pickup (0.351), 2006 Toyota Tacoma (0.335), 1993 Nissan Model not identified (0.325)
- Expected evidence found: Pickup, Asking price
- Performance: retrieval 0.01s; generation 4.34s
- Qualitative assessment: The answer identifies a retrieved pickup below $15,000 with the correct asking price and seller condition label, although its wording is awkward.

**Generated answer**

Based on the provided information, the 2014 Not identified Karen do WD Pickup is listed for $13,900, which falls within the range of $15,000-$16,000. The most relevant details are its price ($13,900) and its condition (Used).

Grounding references used by the local LLM: [S1] [S2] [S3]

### q6: Show diesel vehicles for sale in Phnom Penh.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: NEEDS_REVIEW
- Top evidence: Toyota Land Cruiser Prado (0.506), 2022 Ford Ranger Wildtrak (0.414), 2022 Ford Ranger Wildtrak (0.414)
- Expected evidence found: Diesel, Phnom Penh
- Performance: retrieval 0.01s; generation 7.32s
- Qualitative assessment: The diesel specifications and Phnom Penh locations are supported, but saying a listing is currently available is too strong for a time-stamped marketplace snapshot.

**Generated answer**

[Vehicle: Toyota Land Cruiser Prado] is available for sale in Phnom Penh, priced at $29,300, which is in the used condition. It has a 4-cylinder 2.8-liter turbo diesel engine and a displacement of 2.8 liters. The vehicle's seating capacity is 7, and it comes equipped with a 6-speed automatic transmission. [Vehicle: 2022 Ford Ranger Wildtrak] is also available for sale in Phnom Penh, priced at $33,500, which is in the used condition. It has a 4-cylinder 2.0-liter bi-turbo diesel engine and a displacement of 2.0 liters. The vehicle's seating capacity is 5, and it comes equipped with a 10-speed automatic transmission. [Vehicle: 2022 Ford Ranger Wildtrak] is available for sale in Chroy Changvar, Phnom Penh, and its economy is 8.9–10.7 km/L in official regional diesel

Grounding references used by the local LLM: [S1] [S2] [S3]

### q7: Compare used Toyota and Lexus cars under $20,000.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: FAIL
- Top evidence: 2022 Lexus RX (0.531), 2008 Toyota Prius (0.454), 2005 Lexus RX (0.511)
- Expected evidence found: Toyota, Lexus, Used
- Performance: retrieval 0.01s; generation 6.72s
- Qualitative assessment: The local model invents vehicle abbreviations and unsupported hybrid/manual-transmission comparisons. Retrieval covered both makes, but generation quality is unacceptable.

**Generated answer**

[LSX] and [RX] are both used 2022 models in the $11,000 range, with [LSX] having a slightly higher economy rating of 21-30 mpg combined (EPA). Both have 3.5L displacement, 6-cylinder engine, and 21-30 mpg combined (EPA), indicating they are comparable in terms of fuel efficiency. However, [LSX] is more likely to be a hybrid model due to its higher fuel economy, while [RX] is an SUV with a lower fuel economy but better fuel efficiency. [LSX] is also more likely to be a manual transmission, whereas [RX] has an automatic transmission. Both vehicles are located in Chbar Ampov, Phnom Penh, and have a fuel efficiency rating of 21-30 mpg combined (EPA).

Grounding references used by the local LLM: [S1] [S2] [S3]

### q8: Show 2012 hybrid cars under $18,000.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: PASS
- Top evidence: 2012 Toyota Prius (0.351), 2012 Toyota Prius (0.363)
- Expected evidence found: 2012, Hybrid, Asking price
- Performance: retrieval 0.01s; generation 3.36s
- Qualitative assessment: The answer is concise and accurately reports a retrieved 2012 Prius, its price, hybrid powertrain, efficiency, seating, and transmission.

**Generated answer**

[Toyota Prius] - Price: $15,600, Condition: Used, Body: Hatchback, Fuel: Hybrid, Location: Tuol Kouk, Phnom Penh, Economy: 48 mpg combined, Cylinders: 4, Displacement: 1.8 L, Seats: 5, Transmission: e-CVT.

Grounding references used by the local LLM: [S1] [S2]

### q9: Show sports cars under $50,000.

- Retrieval: PASS
- Citation grounding: PASS
- Manual answer quality: PASS
- Top evidence: 2015 Mercedes-Benz CLA-Class (0.289), 2014 Mercedes-Benz CLA-Class (0.306), 2015 Mercedes-Benz CLA-Class (0.282)
- Expected evidence found: Sports, Asking price
- Performance: retrieval 0.01s; generation 3.85s
- Qualitative assessment: The answer recommends a retrieved sports-classified listing below $50,000 and stays within the available price evidence, though it could compare more than one result.

**Generated answer**

[SA1] The 2015 Mercedes-Benz CLA-Class is priced at $32,000, which falls under the $50,000 range, making it a suitable choice for those looking to purchase a sports car under $50,000.

Grounding references used by the local LLM: [S1] [S2] [S3]

## Limitations and interpretation

- The automatic retrieval metric checks expected evidence coverage, not whether the first result is always the best possible vehicle.
- Citation validation confirms that labels point to supplied chunks; manual review is still required to detect subtle paraphrase errors.
- Khmer24 prices, condition labels, and availability are time-sensitive seller claims. Technical specifications are model-year references and may not match an imported trim exactly.
- Crash safety, reliability, mechanical condition, legal ownership, and fair market value remain outside the evidence collection.
