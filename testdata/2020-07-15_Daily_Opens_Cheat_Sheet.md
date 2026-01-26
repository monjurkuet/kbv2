# Daily Opens Cheat Sheet

## Textual Hierarchy & Theory

### Chart Champions Daily Opens Strategies

> **Pro Tip:** The strategies described are based on price action at the daily open, focusing on volume confirmation and initial balance (IB) characteristics.

> **Pro Tip:** All strategies were explained in the Contenders Daily opens stream on 15.7.2020 at specific timestamps referenced for each setup.

> **Warning:** The "VA" (Value Area) mentioned in the Open Drive strategy is referenced but not explicitly defined in this cheat sheet.

> **Pro Tip:** Volume confirmation is critical across all three strategies - it serves as validation for the trade setups.

**Definition: IB (Initial Balance)** - The price range established during the first hour of trading, used as a reference for the day's price action. The cheat sheet references "strong IB" and "weak IB" as indicators of market conviction.

## The "Visual Decoder"

### 📊 Chart Analysis: Open Drive Strategy

**Metadata:** 
- Timeframe: Not explicitly stated (likely intraday)
- Asset: Not specified
- Indicator Parameters: Value Area (VA) referenced but parameters not provided

**Spatial Annotation Map:**
- "Open outside of VA" label points to price action that opens beyond the previous day's value area
- "Strong IB" annotation appears near the initial trading range of the day
- "Volume" indicator is shown increasing during the directional move

**The Narrative Flow (Chronological):**
Step 1: Price opens outside of the previous day's Value Area (VA). Step 2: Price continues in the opening direction quickly. Step 3: Volume increases during the directional move. Step 4: The Initial Balance (IB) shows a strong push, indicating institutional participation.

**Hard Data Extraction:**
- Reference timestamp: 15.7.2020 (30:41)
- Price action: Opening outside VA
- Volume: Increasing during directional move

### 📊 Chart Analysis: Open Test Drive Strategy

**Metadata:**
- Timeframe: Not explicitly stated (likely intraday)
- Asset: Not specified
- Indicator Parameters: None specified beyond daily open reference

**Spatial Annotation Map:**
- "Weak IB" label points to the initial trading range with low volume
- "Daily Open" reference point is marked on the chart
- "Retest" label points to price returning to the daily open level
- "Volume increase" annotation appears at the retest point

**The Narrative Flow (Chronological):**
Step 1: Price opens with low volume, creating a weak Initial Balance. Step 2: Price moves slowly away from the open. Step 3: Price tests (retraces to) the daily open level. Step 4: Rejection occurs at the daily open level with increasing volume.

**Hard Data Extraction:**
- Reference timestamp: 15.7.2020 (37:11)
- Volume characteristics: Low at open, increases during retest
- Price action: Initial slow movement followed by retest of daily open

### 📊 Chart Analysis: Open Rejection Reverse Strategy

**Metadata:**
- Timeframe: Not explicitly stated (likely intraday)
- Asset: Not specified
- Indicator Parameters: None specified beyond key level reference

**Spatial Annotation Map:**
- "Weak directional move" label points to initial price movement after open
- "Key level" marker shows a significant price point being tested
- "Rejection" annotation appears at the point where price reverses from the key level
- "Volume increase" label is positioned at the reversal point

**The Narrative Flow (Chronological):**
Step 1: Price makes a weak directional move from the open. Step 2: Price tests a key level (unspecified in text). Step 3: Strong rejection occurs at the key level. Step 4: Price reverses direction with increasing volume.

**Hard Data Extraction:**
- Reference timestamp: 15.7.2020 (45:38)
- Volume characteristics: Increases during the reversal
- Price action: Weak initial move, test of key level, strong rejection

## Strategic Logic (Pseudocode Conversion)

```yaml
Strategy Name: Open Drive
Direction: Trend Following (Long/Short depending on direction)
Conditions:
  - IF: Price opens outside previous day's Value Area (VA)
  - AND: Price continues quickly in the opening direction
  - AND: Volume increases during the directional move
  - AND: Initial Balance (IB) shows strong directional push
Triggers:
  - ENTRY: Break of initial consolidation after open with volume confirmation
  - STOP LOSS: Beyond opposite side of Initial Balance (IB)
  - TAKE PROFIT: Measured move equal to IB height or until volume dries up
```

```yaml
Strategy Name: Open Test Drive
Direction: Counter-trend (Opposite to initial move)
Conditions:
  - IF: Daily open occurs with low volume (weak IB)
  - AND: Price initially moves slowly away from open
  - AND: Price retraces to test daily open level
  - AND: Volume increases during the retest
  - AND: Clear rejection pattern forms at daily open level
Triggers:
  - ENTRY: After price rejects daily open level with volume confirmation
  - STOP LOSS: Beyond the tested daily open level
  - TAKE PROFIT: Measured move equal to initial move away from open
```

```yaml
Strategy Name: Open Rejection Reverse
Direction: Counter-trend (Opposite to initial move)
Conditions:
  - IF: Price makes weak directional move from open
  - AND: Price tests a key level (support/resistance)
  - AND: Strong rejection occurs at key level
  - AND: Volume increases during the rejection
  - AND: Clear reversal pattern forms (e.g., pin bar, engulfing)
Triggers:
  - ENTRY: After confirmation of rejection pattern with volume
  - STOP LOSS: Beyond the tested key level
  - TAKE PROFIT: Measured move equal to initial move toward key level
```

## Mathematical & Tabular Data

No explicit mathematical formulas or structured tables were present in the provided cheat sheet. The strategies rely on qualitative price action analysis with volume confirmation rather than quantitative calculations.