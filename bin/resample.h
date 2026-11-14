//! A streaming samplerate converter for interleaved 16-bit audio, used by the CLI's --resample option.
//!
//! Arbitrary ratios (from 1:256 to 256:1) are handled with a windowed-sinc filter (Kaiser window, about -90 dB
//! stopband): the kernel is sampled into a table once and evaluated with linear interpolation, so any input/output
//! ratio works without per-ratio filter design. Resampling changes only the sample grid, never the speed: the output
//! duration matches the input within half an output frame, and the pitch is untouched.
//!
//! Usage: resample_init, then resample_push input frames and resample_pull output frames until the input ends,
//! resample_finish once it did, and pull until it returns 0. resample_free releases the buffer.

#ifndef RESAMPLE_H
#define RESAMPLE_H

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define RESAMPLE_MAX_RATIO 256      //!< Supported range of input:output and output:input ratios
#define RESAMPLE_MAX_CHANNELS 8     //!< QOAMAX_MAX_CHANNEL_COUNT
#define RESAMPLE_KERNEL_HALF 32     //!< Kernel radius in samples of the rate the cutoff refers to
#define RESAMPLE_TABLE_DENSITY 2048 //!< Table entries per kernel sample (linear interpolation in between)

typedef struct {
  int channels;
  uint32_t in_rate, out_rate;
  double step;       //!< Input frames per output frame (in_rate / out_rate)
  double cutoff;     //!< Filter cutoff relative to the input rate: min(1, out_rate / in_rate)
  uint64_t in_fed;   //!< Input frames pushed so far
  uint64_t out_made; //!< Output frames pulled so far
  uint64_t out_target; //!< Total output frames, set by resample_finish
  bool finished;     //!< resample_finish was called: the input is complete, missing samples read as zero
  int16_t *data;     //!< The queued input frames [base, base + len), at data[begin * channels ...]
  uint64_t base;
  size_t begin, len, cap; //!< In frames
} resampler;

#define RESAMPLE_PI 3.14159265358979323846

//! The kernel table: h(x) = sinc(x) * kaiser(x / RESAMPLE_KERNEL_HALF) for x in [0, RESAMPLE_KERNEL_HALF].
static float resample_table[RESAMPLE_KERNEL_HALF * RESAMPLE_TABLE_DENSITY + 2];
static bool resample_table_ready;

static double resample_bessel_i0(double x) {
  double sum = 1, term = 1, y = x * x / 4;
  for (int k = 1; k < 64; k++) {
    term *= y / ((double)k * k);
    sum += term;
    if (term < sum * 1e-13) break;
  }
  return sum;
}

static void resample_table_build(void) {
  if (resample_table_ready) return;
  const double beta = 9.0; // a stopband of about -90 dB
  double norm             = resample_bessel_i0(beta);
  for (size_t i = 0; i <= RESAMPLE_KERNEL_HALF * RESAMPLE_TABLE_DENSITY + 1; i++) {
    double x      = (double)i / RESAMPLE_TABLE_DENSITY;
    double sinc   = x == 0 ? 1.0 : sin(RESAMPLE_PI * x) / (RESAMPLE_PI * x);
    double u      = x / RESAMPLE_KERNEL_HALF;
    double window = u >= 1 ? 0.0 : resample_bessel_i0(beta * sqrt(1 - u * u)) / norm;
    resample_table[i] = (float)(sinc * window);
  }
  resample_table_ready = true;
}

//! The kernel at distance x (in samples of the rate the cutoff refers to), with linear interpolation.
static double resample_kernel(double x) {
  if (x > RESAMPLE_KERNEL_HALF) x = RESAMPLE_KERNEL_HALF;
  double pos  = x * RESAMPLE_TABLE_DENSITY;
  size_t i    = (size_t)pos;
  double frac = pos - (double)i;
  return resample_table[i] + frac * ((double)resample_table[i + 1] - resample_table[i]);
}

//! Initializes the resampler. Returns false if the ratio is outside 1:RESAMPLE_MAX_RATIO..RESAMPLE_MAX_RATIO:1.
static bool resample_init(resampler *r, int channels, uint32_t in_rate, uint32_t out_rate) {
  if ((uint64_t)in_rate * RESAMPLE_MAX_RATIO < out_rate || (uint64_t)out_rate * RESAMPLE_MAX_RATIO < in_rate)
    return false;
  resample_table_build();
  *r = (resampler){
      .channels = channels,
      .in_rate  = in_rate,
      .out_rate = out_rate,
      .step     = (double)in_rate / out_rate,
      .cutoff   = out_rate < in_rate ? (double)out_rate / in_rate : 1.0,
  };
  return true;
}

static void resample_free(resampler *r) {
  free(r->data);
  r->data = NULL;
}

//! The exact number of output frames for the given number of input frames (rounded to nearest). The duration error
//! against in_frames / in_rate seconds is at most half an output frame.
static uint64_t resample_output_frames(const resampler *r, uint64_t in_frames) {
  return (in_frames * r->out_rate + r->in_rate / 2) / r->in_rate;
}

//! Queues input frames. Returns false if the buffer couldn't grow (out of memory).
static bool resample_push(resampler *r, const int16_t *frames, size_t count) {
  if (count == 0) return true;
  if (r->begin + r->len + count > r->cap) {
    if (r->begin > 0) {
      memmove(r->data, r->data + r->begin * r->channels, r->len * r->channels * sizeof *frames);
      r->begin = 0;
    }
    if (r->len + count > r->cap) {
      size_t cap = r->cap ? r->cap : 4096;
      while (cap < r->len + count) cap *= 2;
      int16_t *data = realloc(r->data, cap * r->channels * sizeof *data);
      if (!data) return false;
      r->data = data;
      r->cap  = cap;
    }
  }
  memcpy(r->data + (r->begin + r->len) * r->channels, frames, count * r->channels * sizeof *frames);
  r->len += count;
  r->in_fed += count;
  return true;
}

//! No more input follows: the remaining output frames are produced, reading zeros past the input's end.
static void resample_finish(resampler *r) {
  r->finished   = true;
  r->out_target = resample_output_frames(r, r->in_fed);
}

static int16_t resample_at(const resampler *r, int64_t index, int channel) {
  if (index < (int64_t)r->base || index >= (int64_t)(r->base + r->len)) return 0; // zero padding at the edges
  return r->data[(r->begin + (size_t)(index - (int64_t)r->base)) * r->channels + channel];
}

//! Produces up to `capacity` output frames. Returns fewer (or 0) when it needs more input, or once the finished
//! stream is drained.
static size_t resample_pull(resampler *r, int16_t *out, size_t capacity) {
  double radius   = RESAMPLE_KERNEL_HALF / r->cutoff; // kernel radius in input frames
  size_t produced = 0;
  while (produced < capacity) {
    if (r->finished && r->out_made >= r->out_target) break;
    double t     = r->out_made * r->step; // the output frame's position in input frames
    int64_t last = (int64_t)(t + radius);
    if (!r->finished && (uint64_t)last >= r->in_fed) break; // needs input up to index `last`

    int64_t first = (int64_t)ceil(t - radius);
    // Frames before `first` are never needed again (t only grows): drop them.
    if (first > (int64_t)r->base) {
      uint64_t drop = (uint64_t)first - r->base;
      if (drop > r->len) drop = r->len;
      r->begin += drop;
      r->len -= drop;
      r->base += drop;
    }

    double sums[RESAMPLE_MAX_CHANNELS] = {0};
    for (int64_t i = first; i <= last; i++) {
      double weight = resample_kernel(fabs(t - (double)i) * r->cutoff);
      for (int ch = 0; ch < r->channels; ch++) sums[ch] += resample_at(r, i, ch) * weight;
    }
    for (int ch = 0; ch < r->channels; ch++) {
      long value = lrint(sums[ch] * r->cutoff);
      if (value > INT16_MAX) value = INT16_MAX;
      if (value < INT16_MIN) value = INT16_MIN;
      out[produced * r->channels + ch] = (int16_t)value;
    }
    r->out_made++;
    produced++;
  }
  return produced;
}

#endif
