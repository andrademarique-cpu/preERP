/*
 * potentiometer_logger.ino
 *
 * Streams two potentiometer channels over USB serial as CSV, one line per
 * sample:
 *
 *     <micros>,<a0>,<a1>\n
 *
 *   micros : uint32, microseconds since boot (wraps every ~71.6 min; the
 *            host unwraps it)
 *   a0     : raw ADC counts on A0 -> theta1
 *   a1     : raw ADC counts on A1 -> theta2
 *
 * Counts are left raw on purpose: the mapping counts -> degrees is
 * calibration, and calibration belongs on the host where it can be changed
 * without reflashing.
 *
 * Companion host app: notebooks/Potenctiometerlogging.py
 */

const uint8_t  PIN_THETA1 = A0;
const uint8_t  PIN_THETA2 = A1;

const uint32_t BAUD       = 115200;
const uint16_t SAMPLE_HZ  = 200;   // 200 Hz * ~20 B/line = 4 kB/s, well under 11.5 kB/s
const uint8_t  OVERSAMPLE = 4;     // averaged reads per sample, cuts ADC noise ~2x

const uint32_t PERIOD_US = 1000000UL / SAMPLE_HZ;
uint32_t next_us = 0;

/* Average OVERSAMPLE reads of `pin`, in raw ADC counts.
 *
 * The first read after a mux switch is discarded: the sample-and-hold cap
 * needs time to settle to the new channel, and keeping that read couples
 * A0 and A1 into each other. */
uint16_t readAveraged(uint8_t pin) {
  analogRead(pin);
  uint16_t acc = 0;
  for (uint8_t i = 0; i < OVERSAMPLE; ++i) {
    acc += analogRead(pin);
  }
  return acc / OVERSAMPLE;
}

void setup() {
  Serial.begin(BAUD);
  while (!Serial) {
    ;  // needed on native-USB boards (Leonardo/Micro/32u4); no-op on Uno/Nano
  }
  next_us = micros();
}

void loop() {
  // Signed comparison so the schedule survives the micros() rollover.
  if ((int32_t)(micros() - next_us) < 0) {
    return;
  }
  next_us += PERIOD_US;

  const uint32_t t  = micros();
  const uint16_t a0 = readAveraged(PIN_THETA1);
  const uint16_t a1 = readAveraged(PIN_THETA2);

  Serial.print(t);
  Serial.print(',');
  Serial.print(a0);
  Serial.print(',');
  Serial.println(a1);
}
