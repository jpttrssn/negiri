// SPDX-License-Identifier: GPL-3.0-or-later

//! Minimal EXIF writing helpers for frame exports. A hand-rolled little-endian
//! TIFF blob carries the `DateTimeOriginal` (+ optional `DateTimeDigitized`) tags
//! and is spliced into JPEG streams as an APP1 (Exif) segment; PNG writes the
//! same blob through the `png` crate's `eXIf` chunk API. No new crates are
//! introduced — `kamadak-exif` stays read-only and is used here only in the
//! round-trip tests.

/// How many days `month` (1-based) occupies in the given `year`, accounting for
/// leap years.
const fn days_in_month(year: u32, month: u32) -> u32 {
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if leap => 29,
        _ => 28,
    }
}

/// Parses a calendar-valid ISO `YYYY-MM-DD` into `(year, month, day)`, rejecting
/// malformed or out-of-range values. Exactly 10 bytes with hyphen separators are
/// expected (the manifest validates this shape before the value reaches us).
fn parse_iso_date(iso: &str) -> Option<(u32, u32, u32)> {
    if iso.len() != 10 {
        return None;
    }
    let mut parts = iso.splitn(3, '-');
    let year: u32 = parts.next()?.parse().ok()?;
    let month: u32 = parts.next()?.parse().ok()?;
    let day: u32 = parts.next()?.parse().ok()?;
    // Reject non-zero-padded input so we can safely slice later.
    if format!("{year:04}-{month:02}-{day:02}") != iso {
        return None;
    }
    if !(1..=12).contains(&month) || day == 0 || day > days_in_month(year, month) {
        return None;
    }
    Some((year, month, day))
}

/// Whether `s` is the exact EXIF datetime shape `YYYY:MM:DD HH:MM:SS` — 19
/// ASCII chars, colons at 4/7/13/16, a space between the date and time parts,
/// and digits everywhere else.
#[must_use]
fn is_exif_datetime(s: &str) -> bool {
    s.len() == 19
        && s.as_bytes()
            .iter()
            .enumerate()
            .all(|(i, b)| match i {
                4 | 7 | 13 | 16 => *b == b':',
                10 => *b == b' ',
                _ => b.is_ascii_digit(),
            })
}

/// `YYYY-MM-DD` plus `offset_secs` from midnight, as the EXIF
/// `YYYY:MM:DD HH:MM:SS` string that makes a frame's `DateTimeOriginal` when the
/// enclosing roll carries a start date. Returns `None` when `start_iso` is
/// unparseable.
#[must_use]
pub(crate) fn shifted_datetime(start_iso: &str, offset_secs: usize) -> Option<String> {
    let (mut year, mut month, mut day) = parse_iso_date(start_iso)?;
    let day_carry = u32::try_from(offset_secs / 86_400).ok()?;
    let secs = offset_secs % 86_400;
    day += day_carry;
    while day > days_in_month(year, month) {
        day -= days_in_month(year, month);
        month += 1;
        if month > 12 {
            month = 1;
            year += 1;
        }
    }
    let (hh, mm, ss) = (secs / 3600, (secs % 3600) / 60, secs % 60);
    Some(format!(
        "{year:04}:{month:02}:{day:02} {hh:02}:{mm:02}:{ss:02}"
    ))
}

/// Pushes a single little-endian `u16` into `buf`.
fn push_u16(buf: &mut Vec<u8>, value: u16) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Pushes a single little-endian `u32` into `buf`.
fn push_u32(buf: &mut Vec<u8>, value: u32) {
    buf.extend_from_slice(&value.to_le_bytes());
}

/// Writes a single ASCII tag entry into the Exif IFD: `tag`, type 2 (ASCII),
/// count of 20 (chars including the NUL terminator), then the byte offset at
/// which the 20-byte string payload lives within the TIFF blob.
fn push_ascii_entry(buf: &mut Vec<u8>, tag: u16, value_offset: u32) {
    push_u16(buf, tag);
    push_u16(buf, 2);
    push_u32(buf, 20);
    push_u32(buf, value_offset);
}

/// Serializes the minimal little-endian TIFF that a reader resolves into the
/// given `DateTimeOriginal` (and optional `DateTimeDigitized`) in the Exif IFD.
///
/// The layout is 8-byte TIFF header → IFD0 (one entry: the `ExifIFDPointer`) →
/// Exif IFD (one or two ASCII tags) → fixed 20-byte NUL-terminated strings.
///
/// Both `original` and `digitized` must carry the exact EXIF datetime shape
/// `YYYY:MM:DD HH:MM:SS` (19 ASCII chars). Anything else (including kamadak's
/// hyphen-normalized display form) is rejected as `None`.
#[must_use]
pub(crate) fn build_tiff(original: &str, digitized: Option<&str>) -> Option<Vec<u8>> {
    if !is_exif_datetime(original) || digitized.is_some_and(|d| !is_exif_datetime(d)) {
        return None;
    }
    let entries = 1 + u32::from(digitized.is_some());
    // IFD0 sits at byte 8 (right after the header) and holds exactly one entry,
    // which is the ExifIFDPointer pointing at the Exif IFD.  The Exif IFD's
    // entries start immediately after its own 2-byte count + 4-byte next-IFD
    // terminator; the 20-byte ASCII payloads follow that tail.
    let exif_ifd: u32 = 8 + 2 + 12 + 4;   // = 26
    let values: u32 = exif_ifd + 2 + entries * 12 + 4;

    let mut out = Vec::with_capacity(values as usize + entries as usize * 20);

    // TIFF header: little-endian mark, 42, then the first-IFD byte offset.
    out.extend_from_slice(b"II*\0");
    push_u32(&mut out, 8);

    // IFD0: one entry (ExifIFDPointer 0x8769, LONG), then next-IFD = 0.
    push_u16(&mut out, 1);
    push_u16(&mut out, 0x8769);
    push_u16(&mut out, 4);
    push_u32(&mut out, 1);
    push_u32(&mut out, exif_ifd);
    push_u32(&mut out, 0);

    // Exif IFD: DateTimeOriginal always; DateTimeDigitized when available.
    push_u16(&mut out, u16::try_from(entries).ok()?);
    push_ascii_entry(&mut out, 0x9003, values);
    if digitized.is_some() {
        push_ascii_entry(&mut out, 0x9004, values + 20);
    }
    push_u32(&mut out, 0);

    // 20-byte ASCII payloads (NUL-terminated strings).
    out.extend_from_slice(original.as_bytes());
    out.push(0);
    if let Some(d) = digitized {
        out.extend_from_slice(d.as_bytes());
        out.push(0);
    }
    Some(out)
}

/// Builds a complete JPEG APP1 (Exif) segment ready to splice into a JPEG byte
/// stream: the `FF E1` marker, a big-endian length of the whole segment (6-byte
/// `Exif\0\0` prefix + the TIFF blob), the prefix, and the TIFF.
#[must_use]
pub(crate) fn jpeg_app1(tiff: &[u8]) -> Vec<u8> {
    let mut app1 = Vec::with_capacity(8 + tiff.len());
    app1.extend_from_slice(&[0xFF, 0xE1]);
    // The length field counts the segment bytes after the marker (the 2 length
    // bytes themselves, the 6-byte `Exif\0\0` prefix, and the TIFF), so
    // `8 + tiff.len()`. The blob is well under u16::MAX.
    let len = u16::try_from(tiff.len() + 8).unwrap_or(u16::MAX);
    app1.extend_from_slice(&len.to_be_bytes());
    app1.extend_from_slice(b"Exif\0\0");
    app1.extend_from_slice(tiff);
    app1
}

/// Inserts `segment` immediately after the JPEG SOI (byte-order) marker — a
/// valid position where any reader walks marker segments.  Returns the stream
/// unchanged when it does not start with `FF D8`.
#[must_use]
pub(crate) fn splice_after_soi(stream: &[u8], segment: &[u8]) -> Vec<u8> {
    if stream.len() < 2 || stream[..2] != [0xFF, 0xD8] {
        return stream.to_vec();
    }
    let mut out = Vec::with_capacity(stream.len() + segment.len());
    out.extend_from_slice(&stream[..2]);
    out.extend_from_slice(segment);
    out.extend_from_slice(&stream[2..]);
    out
}

/// Reads `file_name`'s `DateTimeOriginal` (in `dir`) as its **raw** EXIF text
/// `YYYY:MM:DD HH:MM:SS` — the digitization moment for a scanned negative —
/// rather than kamadak's display-normalized (hyphen) form, so the value can be
/// embedded verbatim in an exported blob. Returns `None` when the file lacks
/// the tag, the value is not ASCII text, or it is not the exact datetime shape.
pub(crate) fn raw_datetime_original(dir: &std::path::Path, file_name: &str) -> Option<String> {
    let file = std::fs::File::open(dir.join(file_name)).ok()?;
    let data = exif::Reader::new()
        .read_from_container(&mut std::io::BufReader::new(file))
        .ok()?;
    let field = data.get_field(exif::Tag::DateTimeOriginal, exif::In::PRIMARY)?;
    let exif::Value::Ascii(parts) = &field.value else {
        return None;
    };
    let mut bytes = parts.concat();
    // Some carriers include the NUL terminator; kamadak does not always strip
    // it, so drop a single trailing one before validating the shape.
    if bytes.last() == Some(&0) {
        bytes.pop();
    }
    let text = std::str::from_utf8(&bytes).ok()?;
    is_exif_datetime(text).then(|| text.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use exif::{In, Reader, Tag, Value};
    use std::io::Cursor;

    /// Helper: read back a tag from a parsed EXIF blob as its raw ASCII bytes
    /// (including the NUL terminator, matching the 20-byte embedded payload).
    fn ascii_bytes(exif_data: &exif::Exif, tag: Tag) -> Vec<u8> {
        let field = exif_data
            .get_field(tag, In::PRIMARY)
            .unwrap_or_else(|| panic!("missing tag {tag:?}"));
        match &field.value {
            Value::Ascii(parts) => parts.concat(),
            other => panic!("unexpected value {other:?}"),
        }
    }

    #[test]
    fn shifted_datetime_zero() {
        assert_eq!(
            shifted_datetime("2024-05-09", 0).unwrap(),
            "2024:05:09 00:00:00"
        );
    }

    #[test]
    fn shifted_datetime_end_of_day() {
        assert_eq!(
            shifted_datetime("2024-05-09", 86_399).unwrap(),
            "2024:05:09 23:59:59"
        );
    }

    #[test]
    fn shifted_datetime_day_rollover() {
        assert_eq!(
            shifted_datetime("2024-05-09", 86_400).unwrap(),
            "2024:05:10 00:00:00"
        );
    }

    #[test]
    fn shifted_datetime_month_rollover() {
        assert_eq!(
            shifted_datetime("2024-01-31", 86_400).unwrap(),
            "2024:02:01 00:00:00"
        );
    }

    #[test]
    fn shifted_datetime_year_rollover() {
        assert_eq!(
            shifted_datetime("2024-12-31", 86_400).unwrap(),
            "2025:01:01 00:00:00"
        );
    }

    #[test]
    fn shifted_datetime_leap_year() {
        assert_eq!(
            shifted_datetime("2024-02-28", 86_400).unwrap(),
            "2024:02:29 00:00:00"
        );
    }

    #[test]
    fn shifted_datetime_non_leap_year() {
        assert_eq!(
            shifted_datetime("2023-02-28", 86_400).unwrap(),
            "2023:03:01 00:00:00"
        );
    }

    #[test]
    fn shifted_datetime_arbitrary_offset() {
        assert_eq!(
            shifted_datetime("2024-05-09", 3_661).unwrap(),
            "2024:05:09 01:01:01"
        );
    }

    #[test]
    fn shifted_datetime_invalid_month() {
        assert!(shifted_datetime("2024-13-01", 0).is_none());
    }

    #[test]
    fn shifted_datetime_invalid_day() {
        assert!(shifted_datetime("2024-05-32", 0).is_none());
    }

    #[test]
    fn shifted_datetime_wrong_format() {
        assert!(shifted_datetime("12/05/2024", 0).is_none());
        assert!(shifted_datetime("2024-5-9", 0).is_none());
    }

    #[test]
    fn shifted_datetime_unpadded() {
        assert!(shifted_datetime("2024-05-09", 0).is_some());
        assert!(shifted_datetime("2024-05-9", 0).is_none());
    }

    #[test]
    fn build_tiff_original_only_round_trip() {
        let tiff = build_tiff("2024:05:09 00:00:12", None).unwrap();
        let parsed = exif::Reader::new()
            .read_from_container(&mut Cursor::new(&tiff))
            .unwrap();
        assert_eq!(
            ascii_bytes(&parsed, exif::Tag::DateTimeOriginal),
            b"2024:05:09 00:00:12"
        );
    }

    #[test]
    fn build_tiff_both_tags_round_trip() {
        let tiff = build_tiff("2024:05:09 00:00:12", Some("2024:05:10 14:30:00")).unwrap();
        let parsed = exif::Reader::new()
            .read_from_container(&mut Cursor::new(&tiff))
            .unwrap();
        assert_eq!(
            ascii_bytes(&parsed, exif::Tag::DateTimeOriginal),
            b"2024:05:09 00:00:12"
        );
        assert_eq!(
            ascii_bytes(&parsed, exif::Tag::DateTimeDigitized),
            b"2024:05:10 14:30:00"
        );
    }

    #[test]
    fn build_tiff_wrong_lengths() {
        assert!(build_tiff("2024:05:09 00:00:1", None).is_none());
        assert!(build_tiff("2024:05:09 00:00:12", Some("too short")).is_none());
    }

    #[test]
    fn build_tiff_rejects_hyphen_normalized_datetime() {
        // kamadak's `display_value()` normalizes EXIF times to `YYYY-MM-DD
        // HH:MM:SS`; that form must not leak into the embedded ASCII value.
        assert!(build_tiff("2024:05:09 00:00:12", None).is_some());
        assert!(
            build_tiff("2024-05-09 00:00:12", None).is_none(),
            "original keeps the colon shape"
        );
        assert!(
            build_tiff("2024:05:09 00:00:12", Some("2024-05-10 12:00:00")).is_none(),
            "digitized keeps the colon shape"
        );
    }

    #[test]
    fn jpeg_round_trip() {
        let mut stream = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut stream, 95);
        encoder
            .encode(&[128u8; 64], 8, 8, jpeg_encoder::ColorType::Luma)
            .unwrap();
        let tiff = build_tiff("2024:05:09 00:00:00", None).unwrap();
        let stamped = splice_after_soi(&stream, &jpeg_app1(&tiff));
        let parsed = exif::Reader::new()
            .read_from_container(&mut Cursor::new(&stamped))
            .unwrap();
        assert_eq!(
            ascii_bytes(&parsed, exif::Tag::DateTimeOriginal),
            b"2024:05:09 00:00:00"
        );
    }

    #[test]
    fn jpeg_round_trip_with_digitized() {
        let mut stream = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut stream, 95);
        encoder
            .encode(&[128u8; 64], 8, 8, jpeg_encoder::ColorType::Luma)
            .unwrap();
        let tiff = build_tiff("2024:05:09 00:00:00", Some("2024:05:10 12:00:00")).unwrap();
        let stamped = splice_after_soi(&stream, &jpeg_app1(&tiff));
        let parsed = exif::Reader::new()
            .read_from_container(&mut Cursor::new(&stamped))
            .unwrap();
        assert_eq!(
            ascii_bytes(&parsed, exif::Tag::DateTimeOriginal),
            b"2024:05:09 00:00:00"
        );
        assert_eq!(
            ascii_bytes(&parsed, exif::Tag::DateTimeDigitized),
            b"2024:05:10 12:00:00"
        );
    }

    #[test]
    fn splice_after_soi_non_jpeg_passthrough() {
        let stream = b"not a jpeg";
        assert_eq!(splice_after_soi(stream, &[0xFF]), stream.to_vec());
    }

    #[test]
    fn raw_datetime_original_reads_colon_form() {
        let mut stream = Vec::new();
        let mut encoder = jpeg_encoder::Encoder::new(&mut stream, 95);
        encoder
            .encode(&[128u8; 16], 4, 4, jpeg_encoder::ColorType::Luma)
            .unwrap();
        let tiff = build_tiff("2024:05:09 00:00:00", None).unwrap();
        let stamped = splice_after_soi(&stream, &jpeg_app1(&tiff));
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "negiri_exif_readback_{}.jpg",
            std::process::id()
        ));
        std::fs::write(&path, &stamped).unwrap();
        let got = raw_datetime_original(&dir, &path.file_name().unwrap().to_string_lossy());
        let _ = std::fs::remove_file(&path);
        // The raw tag text round-trips in `YYYY:MM:DD HH:MM:SS`, not kamadak's
        // hyphen-normalized display form.
        assert_eq!(got, Some("2024:05:09 00:00:00".to_owned()));
    }

    #[test]
    fn png_exif_round_trips_the_raw_tiff() {
        // Mirrors `export_png`: a 16-bit grayscale PNG written via the raw `png`
        // crate with the TIFF blob carried as an `eXIf` chunk before the IDAT,
        // read back through kamadak (which scans PNG chunks for `eXIf`) — the
        // other half of the export stamping, in addition to the JPEG splice.
        let tiff = build_tiff("2024:05:09 18:24:36", Some("2024:05:10 09:00:00")).unwrap();
        let mut png_bytes = Vec::new();
        let mut encoder = png::Encoder::new(&mut png_bytes, 4, 4);
        encoder.set_color(png::ColorType::Grayscale);
        encoder.set_depth(png::BitDepth::Sixteen);
        let mut writer = encoder.write_header().unwrap();
        writer.write_chunk(png::chunk::eXIf, &tiff).unwrap();
        writer.write_image_data(&[0u8; 32]).unwrap();
        drop(writer);

        let parsed = Reader::new()
            .read_from_container(&mut Cursor::new(&png_bytes))
            .unwrap();
        assert_eq!(
            ascii_bytes(&parsed, Tag::DateTimeOriginal),
            b"2024:05:09 18:24:36"
        );
        assert_eq!(
            ascii_bytes(&parsed, Tag::DateTimeDigitized),
            b"2024:05:10 09:00:00"
        );
    }
}
