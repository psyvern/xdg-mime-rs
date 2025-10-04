use nom::branch::alt;
use nom::bytes::complete::{is_a, tag, take, take_until, take_while};
use nom::character::complete::{char, line_ending};
use nom::character::is_hex_digit;
use nom::combinator::{map_res, opt, peek};
use nom::multi::{many0, many1};
use nom::number::complete::be_u16;
use nom::sequence::tuple;
use nom::IResult;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};
use std::str::{self, FromStr};
use std::vec::Vec;

use mediatype::MediaTypeBuf as Mime;

pub fn to_string(s: &[u8]) -> std::result::Result<&str, std::str::Utf8Error> {
    str::from_utf8(s)
}

pub fn to_u32(s: std::result::Result<&str, std::str::Utf8Error>, or_default: u32) -> u32 {
    match s {
        Ok(t) => str::FromStr::from_str(t).unwrap_or(or_default),
        Err(_) => or_default,
    }
}

pub fn buf_to_u32(s: &[u8], or_default: u32) -> u32 {
    to_u32(to_string(s), or_default)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MagicRule {
    indent: u32,
    start_offset: u32,
    value: Vec<u8>,
    mask: Option<Vec<u8>>,
    word_size: u32,
    range_length: u32,
}

fn masked_slices_are_equal(a: &[u8], b: &[u8], mask: &[u8]) -> bool {
    assert!(a.len() == b.len() && a.len() == mask.len());

    let masked_a = a.iter().zip(mask.iter()).map(|(x, m)| *x & *m);
    let masked_b = b.iter().zip(mask.iter()).map(|(x, m)| *x & *m);

    masked_a.eq(masked_b)
}

impl MagicRule {
    pub fn matches_data(&self, data: &[u8]) -> bool {
        assert!(self.mask.is_none() || self.mask.as_ref().unwrap().len() == self.value.len());

        let start = self.start_offset as usize;
        let range_length = self.range_length as usize;
        let value_len = self.value.len();

        let mut data_windows = data.windows(value_len).skip(start).take(range_length);

        match &self.mask {
            Some(mask) => {
                data_windows.any(|data_w| masked_slices_are_equal(data_w, &self.value, mask))
            }

            None => data_windows.any(|data_w| data_w == &self.value[..]),
        }
    }

    pub fn extent(&self) -> usize {
        let value_len = self.value.len();
        let offset = self.start_offset as usize;
        let range_len = self.range_length as usize;

        value_len + offset + range_len
    }
}

// Indentation level, can be 0
fn indent_level(bytes: &[u8]) -> IResult<&[u8], u32> {
    let (bytes, res) = take_until(">")(bytes)?;

    Ok((bytes, buf_to_u32(res, 0)))
}

// Offset, can be 0
fn start_offset(bytes: &[u8]) -> IResult<&[u8], u32> {
    let (bytes, res) = take_until("=")(bytes)?;

    Ok((bytes, buf_to_u32(res, 0)))
}

// <word_size> = '~' (0 | 1 | 2 | 4)
fn word_size(bytes: &[u8]) -> IResult<&[u8], Option<u32>> {
    let alt_size = alt((tag("0"), tag("1"), tag("2"), tag("4")));
    let word_size = tuple((tag("~"), alt_size));
    let (bytes, res) = opt(word_size)(bytes)?;

    let size = match res {
        Some(v) => buf_to_u32(v.1, 1),
        None => return Ok((bytes, None)),
    };

    Ok((bytes, Some(size)))
}

// <range_length> = '+' <u32>
fn range_length(bytes: &[u8]) -> IResult<&[u8], Option<u32>> {
    let range_len = tuple((tag("+"), take_while(is_hex_digit)));
    let (bytes, res) = opt(range_len)(bytes)?;

    let len = match res {
        Some(v) => buf_to_u32(v.1, 1),
        None => return Ok((bytes, None)),
    };

    Ok((bytes, Some(len)))
}

// magic_rule =
// [ <indent> ] '>' <start-offset> '=' <value_length> <value>
// [ '&' <mask> ] [ <word_size> ] [ <range_length> ]
// '\n'

fn value(bytes: &[u8], length: u16) -> IResult<&[u8], Vec<u8>> {
    let (bytes, res) = take(length)(bytes)?;

    Ok((bytes, res.to_vec()))
}

fn mask(bytes: &[u8], length: u16) -> IResult<&[u8], Option<Vec<u8>>> {
    let (bytes, res) = opt(tuple((char('&'), take(length))))(bytes)?;

    let value = match res {
        Some(v) => v.1.to_vec(),
        None => return Ok((bytes, None)),
    };

    Ok((bytes, Some(value)))
}

fn magic_rule(bytes: &[u8]) -> IResult<&[u8], MagicRule> {
    let (bytes, _) = peek(is_a("0123456789>"))(bytes)?;

    let (bytes, _indent) = indent_level(bytes)?;

    let (bytes, _) = tag(">")(bytes)?;
    let (bytes, _start_offset) = start_offset(bytes)?;

    let (bytes, _) = tag("=")(bytes)?;
    let (bytes, _value_length) = be_u16(bytes)?;
    let (bytes, _value) = value(bytes, _value_length)?;
    let (bytes, _mask) = mask(bytes, _value_length)?;

    let (bytes, _word_size) = word_size(bytes)?;
    let (bytes, _range_length) = range_length(bytes)?;

    let (bytes, _) = line_ending(bytes)?;

    Ok((
        bytes,
        MagicRule {
            indent: _indent,
            start_offset: _start_offset,
            value: _value,
            mask: _mask,
            word_size: _word_size.unwrap_or(1),
            range_length: _range_length.unwrap_or(1),
        },
    ))
}

#[derive(Clone, PartialEq, Eq)]
pub struct MagicEntry {
    pub priority: u32,
    pub rules: Vec<MagicRule>,
}

impl fmt::Debug for MagicEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "(priority: {:?}):\nrules:\n{:?}",
            self.priority, self.rules
        )
    }
}

impl MagicEntry {
    pub fn matches(&self, data: &[u8]) -> bool {
        let mut current_level = 0;

        let mut iter = self.rules.iter().peekable();
        while let Some(rule) = iter.next() {
            // The rules are a flat list that represent a tree; the "indent"
            // is the depth of the rule in the tree.
            //
            // Check the rule at the current level
            if rule.indent == current_level && rule.matches_data(data) {
                // If the next rule has a lower level, or it's the last
                // rule, we found our match
                match iter.peek() {
                    Some(next) => {
                        if next.indent <= current_level {
                            return true;
                        }

                        // Otherwise, increase the level and check the
                        // next rule
                        current_level += 1;
                    }
                    None => {
                        // last rule
                        return true;
                    }
                };
            }
        }

        false
    }

    pub fn max_extents(&self) -> usize {
        self.rules.iter().map(MagicRule::extent).max().unwrap_or(0)
    }
}

fn priority(bytes: &[u8]) -> IResult<&[u8], u32> {
    let (bytes, res) = take_until(":")(bytes)?;

    Ok((bytes, buf_to_u32(res, 0)))
}

fn mime_type(bytes: &[u8]) -> IResult<&[u8], Mime> {
    map_res(map_res(take_until("]\n"), str::from_utf8), Mime::from_str)(bytes)
}

// magic_header =
// '[' <priority> ':' <mime_type> ']' '\n'
fn magic_header(bytes: &[u8]) -> IResult<&[u8], (u32, Mime)> {
    let (bytes, (_, _priority, _, _mime_type, _)) =
        tuple((tag("["), priority, tag(":"), mime_type, tag("]\n")))(bytes)?;

    Ok((bytes, (_priority, _mime_type)))
}

// magic_entry =
// <magic_header>
// <magic_rule>+
fn magic_entry(bytes: &[u8]) -> IResult<&[u8], (Mime, MagicEntry)> {
    let (bytes, (_header, _rules)) = tuple((magic_header, many1(magic_rule)))(bytes)?;

    Ok((
        bytes,
        (
            _header.1,
            MagicEntry {
                priority: _header.0,
                rules: _rules,
            },
        ),
    ))
}

fn from_u8_to_entries(bytes: &[u8]) -> IResult<&[u8], Vec<(Mime, MagicEntry)>> {
    let (bytes, (_, entries)) = tuple((tag("MIME-Magic\0\n"), many0(magic_entry)))(bytes)?;

    Ok((bytes, entries))
}

pub fn lookup_data<'a>(
    entries: &'a HashMap<Mime, Vec<MagicEntry>>,
    data: &[u8],
) -> Option<(&'a Mime, u32)> {
    entries.iter().find_map(|(key, value)| {
        value
            .iter()
            .find(|x| x.matches(data))
            .map(|x| (key, x.priority))
    })
}

pub fn max_extents(entries: &HashMap<Mime, Vec<MagicEntry>>) -> usize {
    entries
        .values()
        .flatten()
        .map(|x| x.max_extents())
        .max()
        .unwrap_or(0)
}

pub fn read_magic_from_file<P: AsRef<Path>>(file_name: P) -> Vec<(Mime, MagicEntry)> {
    let mut f = match File::open(file_name.as_ref()) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    let mut magic_buf = Vec::<u8>::new();

    f.read_to_end(&mut magic_buf).unwrap();

    match from_u8_to_entries(magic_buf.as_slice()) {
        Ok(v) => v.1,
        Err(_) => Vec::new(),
    }
}

pub fn read_magic_from_dir<P: AsRef<Path>>(dir: P) -> Vec<(Mime, MagicEntry)> {
    let mut magic_file = PathBuf::new();
    magic_file.push(dir);
    magic_file.push("magic");

    read_magic_from_file(magic_file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nom::HexDisplay;
    use nom::Offset;

    #[test]
    fn parse_magic_header() {
        let res = magic_header("[50:application/x-yaml]\n".as_bytes());

        match res {
            Ok((i, o)) => {
                assert_eq!(i.len(), 0);
                println!("parsed:\n{:?}", o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic rule");
            }
        }
    }

    #[test]
    fn parse_one_magic_rule() {
        let simple = include_bytes!("../test_files/parser/single_rule");
        println!("bytes:\n{}", &simple.to_hex(8));
        let simple_res = magic_rule(simple);

        match simple_res {
            Ok((i, o)) => {
                println!("remaining:\n{}", &i.to_hex_from(8, simple.offset(i)));
                println!("parsed:\n{:?}", o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic rule");
            }
        }

        let range = include_bytes!("../test_files/parser/rule_with_range");
        println!("bytes:\n{}", &range.to_hex(8));
        let range_res = magic_rule(range);

        match range_res {
            Ok((i, o)) => {
                println!("remaining:\n{}", &i.to_hex_from(8, range.offset(i)));
                println!("parsed:\n{:?}", o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic rule");
            }
        }

        let ws = include_bytes!("../test_files/parser/rule_with_ws");
        println!("bytes:\n{}", &ws.to_hex(8));
        let ws_res = magic_rule(ws);

        match ws_res {
            Ok((i, o)) => {
                println!("remaining:\n{}", &i.to_hex_from(8, ws.offset(i)));
                println!("parsed:\n{:?}", o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic rule");
            }
        }
    }

    #[test]
    fn parse_simple_magic_entry() {
        let data = include_bytes!("../test_files/parser/single_entry");
        println!("bytes:\n{}", &data.to_hex(8));
        let res = magic_entry(data);

        match res {
            Ok((i, o)) => {
                println!("remaining:\n{}", &i.to_hex_from(8, data.offset(i)));
                println!("parsed:\n{:?}", o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic entry");
            }
        }
    }

    #[test]
    fn parse_magic_entry() {
        let data = include_bytes!("../test_files/parser/many_rules");
        println!("bytes:\n{}", &data.to_hex(8));
        let res = magic_entry(data);

        match res {
            Ok((i, o)) => {
                println!("remaining:\n{}", &i.to_hex_from(8, data.offset(i)));
                println!("parsed:\n{:?}", o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic entry");
            }
        }
    }

    #[test]
    fn parse_magic_file() {
        let data = include_bytes!("../test_files/mime/magic");
        let res = from_u8_to_entries(data);

        match res {
            Ok((i, o)) => {
                println!("remaining:\n{}", &i.to_hex_from(8, data.offset(i)));
                println!("parsed {} magic entries:\n{:#?}", o.len(), o);
            }
            e => {
                println!("invalid or incomplete: {:?}", e);
                panic!("cannot parse magic file");
            }
        }
    }

    #[test]
    fn magic_rule_matches_data() {
        let rule = MagicRule {
            indent: 0,
            start_offset: 0,
            value: vec![b'h', b'e', b'l', b'l', b'o'],
            mask: None,
            word_size: 1,
            range_length: 30,
        };

        assert!(rule.matches_data(b"hello world"));
        assert!(rule.matches_data(b"world hello"));
    }

    #[test]
    fn magic_rule_matches_data_with_start_offset() {
        let rule = MagicRule {
            indent: 0,
            start_offset: 1,
            value: vec![b'h', b'e', b'l', b'l', b'o'],
            mask: None,
            word_size: 1,
            range_length: 30,
        };

        assert!(!rule.matches_data(b"hello world"));
        assert!(rule.matches_data(b"xhello world"));
        assert!(rule.matches_data(b"world hello"));
    }

    #[test]
    fn magic_rule_matches_data_with_range_length() {
        let rule = MagicRule {
            indent: 0,
            start_offset: 0,
            value: vec![b'h', b'e', b'l', b'l', b'o'],
            mask: None,
            word_size: 1,
            range_length: 10,
        };

        assert!(rule.matches_data(b"hello world"));
        assert!(rule.matches_data(b"12345hello"));
        assert!(rule.matches_data(b"123456789hello"));
        assert!(!rule.matches_data(b"1234567890hello"));
        assert!(!rule.matches_data(b"too long a prefix for this to match hello"));
    }

    #[test]
    fn magic_rule_matches_data_with_start_offset_and_range_length() {
        let rule = MagicRule {
            indent: 0,
            start_offset: 1,
            value: vec![b'h', b'e', b'l', b'l', b'o'],
            mask: None,
            word_size: 1,
            range_length: 3,
        };

        assert!(!rule.matches_data(b"hello world"));
        assert!(rule.matches_data(b"1hello world"));
        assert!(rule.matches_data(b"12hello world"));
        assert!(rule.matches_data(b"123hello world"));
        assert!(!rule.matches_data(b"1234hello world"));
    }

    #[test]
    fn magic_rule_matches_data_with_mask() {
        let rule = MagicRule {
            indent: 0,
            start_offset: 0,
            value: vec![b'h', b'E', b'l', b'l', b'O'],
            mask: Some(vec![!0x20; 5]),
            word_size: 1,
            range_length: 30,
        };

        assert!(rule.matches_data(b"HeLlo world"));
        assert!(rule.matches_data(b"world HeLlo"));
        assert!(rule.matches_data(b"12345heLLO"));
        assert!(!rule.matches_data(b"HuLLO WORLD"));
    }
}
