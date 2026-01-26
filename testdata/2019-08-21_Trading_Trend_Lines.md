# Trading Trend Lines

## Textual Hierarchy & Theory

### Trend Lines

> **Definition: Confirmed Trendline** - A trendline that has held for 3 bounces so far.

> A trendline is really useful for determining the strength of a move. If price breaks below this trend line one can take profit on a long, or open a short position.

> Many people will wait for a trend line to be 'confirmed' before they will take a trade from it. For a trendline to be classed as confirmed it requires 3 touches.

> What can happen is that now the trend line is confirmed, too many people are watching the trendline and it becomes a 'crowded trade' and it is less likely to hold.

> Daniel has his own way of trading trend lines. He buys the third touch of the trendline and shorts the fourth touch if price breaks down and retests it as resistance (longing 4th touch is crowded trade).

### Trend Line Fan Break

> The 3 trend line fan break strategy is best used on HTF (Higher Time Frame).

> One has to select the pivotal high and connect the series of highs that are forming the downtrend. When price breaks through trend line three this can be a confirmation in reversal.

> The first downtrend trendline keeps price in a downtrend during an extended time. Price finally breaks through the first trendline and forms a top, retraces back to trend line 1 resistance which is now support. Price goes through trend line 2 and forms a 3rd top, and retraces back to trend line 2 support.

### 45 Degree Trendline Stops

> This tool measures the strength of the trend. When trend rises/declines at 45-degree angle this is a healthy trend and can continue while parabolic and weak trend are not healthy.

> A trend should rise or decline at an angle of about 45 degrees, this is a healthy trend that can continue (green).

> Too far over that and the move is parabolic and when the price is broken the retracement will be large (blue).

> Too far below that and the trend may not be trusted as it is too weak (red line).

> Internal trendlines can be used to connect as many lows and highs as possible which are respected.

## 📊 Chart Analysis: Trend Line Confirmation Example

**Metadata:**  
- Timeframe: Not explicitly specified (likely intraday based on context)
- Asset: Not explicitly specified
- Indicator Parameters: Not applicable (pure price action)

**Spatial Annotation Map:**
- The chart shows a trendline with 3 bounces that has been holding
- Text labels indicate "I would buy this 3rd touch" on the third contact with the trendline
- A note explains "many would not buy this but instead wait for the 4th touch"
- The chart includes an annotation about entering short "upon price breaking down from the trendline, back testing it as resistance, and then breaking down below the low"

**The Narrative Flow (Chronological):**
Step 1: Price establishes an uptrend with multiple higher highs and higher lows. Step 2: A trendline is drawn connecting at least two significant lows. Step 3: Price bounces from the trendline for the third time. Step 4: Price breaks below the trendline and retests it as resistance. Step 5: Price breaks below the previous low after the retest.

**Hard Data Extraction:**
- Number of touches for "confirmed" trendline: 3
- Number of touches for crowded trade: 4

## 📊 Chart Analysis: Live Trade Example

**Metadata:**  
- Timeframe: Not explicitly specified
- Asset: Not explicitly specified
- Indicator Parameters: Not applicable

**Spatial Annotation Map:**
- The chart shows a trendline with four touchpoints
- A label indicates "People waiting for confirmation and buying the 4th touch lose the trade"
- The chart shows price breaking through the trendline on the fourth touch
- There is a clear retest of the broken trendline as resistance
- The chart shows subsequent downward price movement after the breakdown

**The Narrative Flow (Chronological):**
Step 1: Price forms a trendline with multiple touches. Step 2: Price approaches the trendline for the fourth time. Step 3: Price breaks the trendline support on the fourth touch. Step 4: Price retests the broken trendline as resistance. Step 5: Price moves down for another wave after the retest holds as resistance.

**Hard Data Extraction:**
- Number of touches before breakdown: 4
- Post-breakdown action: Retest of trendline as resistance
- Result: Downward price movement for another wave

## 📊 Chart Analysis: Trend Line Fan Break

**Metadata:**  
- Timeframe: Higher Time Frame (HTF)
- Asset: Not explicitly specified
- Indicator Parameters: Not applicable

**Spatial Annotation Map:**
- The chart shows three trendlines (labeled 1, 2, and 3) forming a fan pattern
- Trendline 1 connects the initial series of highs in a downtrend
- Trendline 2 is drawn after price breaks trendline 1
- Trendline 3 is drawn after price breaks trendline 2
- The chart shows price breaking through trendline 3

**The Narrative Flow (Chronological):**
Step 1: Price establishes a downtrend with multiple lower highs. Step 2: First trendline (Trendline 1) is drawn connecting the series of highs. Step 3: Price breaks through Trendline 1 and forms a top. Step 4: Price retraces back to Trendline 1 (now acting as resistance). Step 5: Price breaks through Trendline 2 and forms a 3rd top. Step 6: Price retraces back to Trendline 2 (now acting as support). Step 7: Price breaks through Trendline 3, confirming a potential reversal.

**Hard Data Extraction:**
- Number of trendlines in fan pattern: 3
- Sequence: Price breaks Trendline 1 → forms top → retests Trendline 1 → breaks Trendline 2 → forms 3rd top → retests Trendline 2 → breaks Trendline 3

## 📊 Chart Analysis: 45 Degree Trendline Stops

**Metadata:**  
- Timeframe: Not explicitly specified
- Asset: Not explicitly specified
- Indicator Parameters: Not applicable

**Spatial Annotation Map:**
- The chart shows three trendlines at different angles
- A 45-degree line is marked with "45°" label
- Green line represents the ideal 45-degree angle trend
- Blue line represents a steeper (parabolic) trend
- Red line represents a shallow (weak) trend

**The Narrative Flow (Chronological):**
Step 1: Price forms a trend. Step 2: The angle of the trend is measured against a 45-degree reference. Step 3: If the trend is at 45 degrees (green), it's considered healthy. Step 4: If the trend is steeper than 45 degrees (blue), it's considered parabolic. Step 5: If the trend is shallower than 45 degrees (red), it's considered weak.

**Hard Data Extraction:**
- Ideal trend angle: 45 degrees
- Parabolic trend: >45 degrees
- Weak trend: <45 degrees

## Strategic Logic

```yaml
Strategy Name: Confirmed Trendline Trading
Direction: Long
Conditions:
  - IF: Trendline has been touched at least 3 times
  - AND: Price is approaching the trendline for the 3rd or 4th touch
Triggers:
  - ENTRY: On 3rd touch (author's preference) or 4th touch (conventional approach)
  - STOP LOSS: Below the trendline with buffer
  - TAKE PROFIT: At previous resistance level or measured move
```

```yaml
Strategy Name: Trendline Break Reversal
Direction: Short
Conditions:
  - IF: Price breaks below an established trendline (with at least 3 touches)
  - AND: Price retests the broken trendline as resistance
  - AND: Price breaks below the most recent swing low
Triggers:
  - ENTRY: On break of most recent swing low after retest
  - STOP LOSS: Above the retested trendline
  - TAKE PROFIT: At next support level or measured move
```

```yaml
Strategy Name: Trend Line Fan Break
Direction: Long (in downtrend reversal context)
Conditions:
  - IF: Price has broken through first trendline in a fan pattern
  - AND: Price has formed a top and retraced to first trendline (now resistance)
  - AND: Price has broken through second trendline
  - AND: Price has formed a third top and retraced to second trendline (now support)
  - AND: Price breaks through third trendline
Triggers:
  - ENTRY: On break of third trendline with confirmation
  - STOP LOSS: Below the third trendline with buffer
  - TAKE PROFIT: At measured move target or next resistance level
```

```yaml
Strategy Name: 45-Degree Trend Strength Filter
Direction: Both (filter for entries)
Conditions:
  - IF: Trend angle is approximately 45 degrees
  - AND: Price is making higher highs/higher lows (uptrend) or lower highs/lower lows (downtrend)
Triggers:
  - ENTRY: Only consider trades in trends with ~45-degree angle
  - STOP LOSS: N/A (this is a filter, not a standalone strategy)
  - TAKE PROFIT: N/A (this is a filter, not a standalone strategy)
```

## Mathematical & Tabular Data

The document doesn't contain specific mathematical formulas or tables that need conversion to LaTeX or Markdown tables. The key mathematical concept is the 45-degree angle reference for trend strength assessment.

The document references a BTCUSD chart from 2014 to 2020 on logscale, but no specific data points are provided in the text extract.