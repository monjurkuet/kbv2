# Knowledge Base: Head and Shoulders Patterns

## Head and Shoulders Top Reversal Pattern

> **Definition: Head and Shoulders Top** - A reversal pattern that forms from three top peaks. The middle peak (the head) rises above the left and right shoulder, which are normally the same height.

> Volume should be highest on the left shoulder rise, lower volume in the rise of the head, and then the lowest volume on the right shoulder rise, thus as the pattern moves along the volume declines.

> A short entry is not taken until the neckline is broken, when the neckline is broken an increase in bearish volume should be seen for confirmation of the break. A trader can enter the short upon the neckline breaking, or wait for a possible retest of the neckline- if the original break was on high sell volume a retest is not as likely, but if a bounce for a retest does occur it should be on a low volume rise before an increase in sell volume from the retest of the neckline to continue downwards.

> **Definition: Measured Move Target** - The price target obtained by taking the price from the top of the head to the neckline, and then taking a copy of the price distance from the head to the trendline, and placing it at the neckline break.

> **Pro Tip:** The neckline break needs to see an increase in volume to confirm the pattern.

### 📊 Chart Analysis: Head and Shoulders Top Pattern

**Metadata:**  
- Timeframe: Not specified in document  
- Asset: Not specified in document  
- Indicator Parameters: Not applicable (pure price pattern)

**Spatial Annotation Map:**  
- The "Head" label is positioned at the highest peak in the center of the pattern  
- "Left Shoulder" label is positioned at the first peak to the left of the head  
- "Right Shoulder" label is positioned at the third peak to the right of the head  
- "Neckline" is drawn connecting the troughs between the shoulders and head  
- "Head height" measurement is shown as a vertical green line from the top of the head to the neckline

**The Narrative Flow (Chronological):**  
Step 1: Price forms left shoulder peak with highest volume.  
Step 2: Price declines to form the first neckline point.  
Step 3: Price rises to form the head (higher peak than shoulders) with lower volume than left shoulder.  
Step 4: Price declines to form the second neckline point.  
Step 5: Price rises to form the right shoulder (similar height to left shoulder) with lowest volume of the three peaks.  
Step 6: Price breaks below the neckline with increased bearish volume.  
Step 7: Price may retest the neckline (now resistance) on low volume before continuing downward.

**Hard Data Extraction:**  
- Head height: [UNCLEAR numerical value]  
- Target calculation: Distance from head top to neckline × 1 (100% projection)  
- Volume sequence: Left shoulder (highest) > Head (medium) > Right shoulder (lowest)

## Inverse Head and Shoulders Pattern

> **Definition: Inverse Head and Shoulders** - A bottom reversal pattern that forms from three bottom peaks. The middle peak (the head) drops below the left and right shoulder, which are normally the same height.

> Sell volume on the left shoulder may be large, followed by less volume on the decline on the head. The rise from the bottom of the head should see an increase in buying volume, than the rise on the left shoulder. The right shoulder should then be on low volume and then in the rise of the right shoulder volume should be increasing.

> A long entry is not taken until the neckline is broken, when the neckline is broken an increase in buy volume needs to be seen for confirmation of the break. A trader can enter the long upon the neckline breaking, or wait for a possible retest of the neckline- if a drop for a retest does occur it should be on a low volume decline before an increase in buy volume from the retest of the neckline to continue upwards.

> **Pro Tip:** The neckline break needs to see an increase in volume to confirm the pattern.

### 📊 Chart Analysis: Inverse Head and Shoulders Bottom Pattern

**Metadata:**  
- Timeframe: Not specified in document  
- Asset: Not specified in document  
- Indicator Parameters: Not applicable (pure price pattern)

**Spatial Annotation Map:**  
- The "Head" label is positioned at the lowest trough in the center of the pattern  
- "Left Shoulder" label is positioned at the first trough to the left of the head  
- "Right Shoulder" label is positioned at the third trough to the right of the head  
- "Neckline" is drawn connecting the peaks between the shoulders and head  
- "Head height" measurement is shown as a vertical green line from the bottom of the head to the neckline

**The Narrative Flow (Chronological):**  
Step 1: Price forms left shoulder trough with significant selling volume.  
Step 2: Price rises to form the first neckline point.  
Step 3: Price declines to form the head (lower trough than shoulders) with decreasing selling volume.  
Step 4: Price rises to form the second neckline point with increasing buying volume.  
Step 5: Price declines to form the right shoulder (similar height to left shoulder) with low volume.  
Step 6: Price rises to break above the neckline with increased bullish volume.  
Step 7: Price may retest the neckline (now support) on low volume before continuing upward.

**Hard Data Extraction:**  
- Head height: [UNCLEAR numerical value]  
- Target calculation: Distance from head bottom to neckline × 1 (100% projection)  
- Volume sequence: Left shoulder (high selling) > Head decline (decreasing selling) > Head recovery (increasing buying) > Right shoulder (low volume)

## Complex Head and Shoulders Patterns

> **Definition: Complex Head and Shoulders** - Variations of the standard pattern that include multiple heads or shoulders. The rules and targets remain the same as standard patterns, but with different combinations.

> You can get complex Inverse H&S bottom reversals and complex H&S tops reversals with two left shoulders and two right shoulders, or one left and right shoulder with two heads (double top) form.

### 📊 Chart Analysis: Complex Head and Shoulders Pattern

**Metadata:**  
- Timeframe: Not specified in document  
- Asset: Not specified in document  
- Indicator Parameters: Not applicable (pure price pattern)

**Spatial Annotation Map:**  
- "Head" label is positioned at the central peak/trough  
- "Left Shoulder 1" and "Left Shoulder 2" labels are positioned at multiple peaks/troughs to the left of the head  
- "Right Shoulder 1" and "Right Shoulder 2" labels are positioned at multiple peaks/troughs to the right of the head  
- "Neckline" is drawn connecting the troughs/peaks between the shoulders and head  
- "Head height" measurement is shown as a vertical green line from the head to the neckline

**The Narrative Flow (Chronological):**  
Step 1: Price forms multiple left shoulder peaks/troughs.  
Step 2: Price forms the head (highest peak or lowest trough).  
Step 3: Price forms multiple right shoulder peaks/troughs.  
Step 4: Price breaks the neckline with confirming volume.  
Step 5: Price may retest the neckline before continuing in the breakout direction.

## Strategic Logic

```yaml
Strategy Name: Head and Shoulders Top Short Setup
Direction: Short
Conditions:
  - IF: Three peak pattern forms with middle peak (head) higher than left and right shoulders
  - AND: Volume sequence shows decreasing volume across the three peaks (left shoulder > head > right shoulder)
  - AND: Clear neckline is established connecting the two troughs between peaks
Triggers:
  - ENTRY: Break of neckline with increased bearish volume
  - ALTERNATIVE ENTRY: Retest of neckline (now resistance) with low volume approach followed by increased selling
  - STOP LOSS: Above right shoulder high (or above head if no clear shoulder formation)
  - TAKE PROFIT: Measured move target = (Head top price - Neckline price) subtracted from neckline break price
```

```yaml
Strategy Name: Inverse Head and Shoulders Bottom Long Setup
Direction: Long
Conditions:
  - IF: Three trough pattern forms with middle trough (head) lower than left and right shoulders
  - AND: Volume sequence shows decreasing selling volume on head decline followed by increasing buying volume on recovery
  - AND: Clear neckline is established connecting the two peaks between troughs
Triggers:
  - ENTRY: Break of neckline with increased bullish volume
  - ALTERNATIVE ENTRY: Retest of neckline (now support) with low volume decline followed by increased buying
  - STOP LOSS: Below right shoulder low (or below head if no clear shoulder formation)
  - TAKE PROFIT: Measured move target = (Neckline price - Head bottom price) added to neckline break price
```

## Mathematical & Tabular Data

**Measured Move Target Calculation:**

For Head and Shoulders Top:
$$Target = Neckline\_Break\_Price - (Head\_Top\_Price - Neckline\_Price)$$

For Inverse Head and Shoulders Bottom:
$$Target = Neckline\_Break\_Price + (Neckline\_Price - Head\_Bottom\_Price)$$

| Pattern Type | Head Height Measurement | Target Calculation Direction |
|-------------|-------------------------|------------------------------|
| Head and Shoulders Top | From head top to neckline | Downward from neckline break |
| Inverse Head and Shoulders Bottom | From head bottom to neckline | Upward from neckline break |

| Volume Pattern | Left Shoulder | Head | Right Shoulder |
|---------------|--------------|------|----------------|
| Head and Shoulders Top | Highest volume | Medium volume | Lowest volume |
| Inverse Head and Shoulders Bottom | High selling volume | Declining selling volume on decline, increasing buying volume on recovery | Low volume on decline, increasing volume on recovery |