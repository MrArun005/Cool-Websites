'use strict'

function clamp (x, lo, hi) {
  if (x < lo) return lo
  if (x > hi) return hi
  return x
}

function inRange (x, lo, hi) {
  return x >= lo && x <= hi
}

function sumTo (n) {
  let s = 0
  for (let i = 1; i <= n; i++) s += i
  return s
}

function maxOf (xs) {
  if (xs.length === 0) return null
  let best = xs[0]
  for (let i = 1; i < xs.length; i++) {
    if (xs[i] > best) best = xs[i]
  }
  return best
}

function indexOfValue (xs, v) {
  for (let i = 0; i < xs.length; i++) {
    if (xs[i] === v) return i
  }
  return -1
}

function isBlank (s) {
  return s.trim().length === 0
}

function repeatWord (word, times, sep) {
  const parts = []
  for (let i = 0; i < times; i++) parts.push(word)
  return parts.join(sep)
}

function percent (part, whole) {
  if (whole === 0) return 0
  return Math.round((part / whole) * 100)
}

function slugify (s) {
  return s.trim().toLowerCase().split(/\s+/).filter((w) => w.length > 0 || false).join('-')
}

module.exports = { clamp, inRange, sumTo, maxOf, indexOfValue, isBlank, repeatWord, percent, slugify }
