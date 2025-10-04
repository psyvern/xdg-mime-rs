use std::fmt;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use mediatype::MediaTypeBuf as Mime;

#[derive(Clone, PartialEq)]
pub struct Alias {
    pub alias: Mime,
    pub mime_type: Mime,
}

impl fmt::Debug for Alias {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Alias {} {}", self.alias, self.mime_type)
    }
}

impl Alias {
    pub fn new(alias: &Mime, mime_type: &Mime) -> Alias {
        Alias {
            alias: alias.clone(),
            mime_type: mime_type.clone(),
        }
    }

    pub fn from_string(s: &str) -> Option<Alias> {
        let mut chunks = s.split_whitespace().fuse();
        let alias = chunks.next().and_then(|s| s.parse().ok())?;
        let mime_type = chunks.next().and_then(|s| s.parse().ok())?;

        // Consume the leftovers, if any
        if chunks.next().is_some() {
            return None;
        }

        Some(Alias { alias, mime_type })
    }

    pub fn is_equivalent(&self, other: &Alias) -> bool {
        self.alias == other.alias
    }
}

pub fn read_aliases_from_file<P: AsRef<Path>>(file_name: P) -> Vec<Alias> {
    let Ok(file) = File::open(file_name) else {
        return Vec::new();
    };

    BufReader::new(&file)
        .lines()
        .map_while(Result::ok)
        .flat_map(|line| {
            if line.is_empty() || line.starts_with('#') {
                return None;
            }

            Alias::from_string(&line)
        })
        .collect()
}

pub fn read_aliases_from_dir<P: AsRef<Path>>(dir: P) -> Vec<Alias> {
    let mut alias_file = PathBuf::new();
    alias_file.push(dir);
    alias_file.push("aliases");

    read_aliases_from_file(alias_file)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_alias() {
        assert!(Alias::new(
            &"application/foo".parse().unwrap(),
            &"application/foo".parse().unwrap()
        )
        .is_equivalent(&Alias::new(
            &"application/foo".parse().unwrap(),
            &"application/x-foo".parse().unwrap()
        )),);
    }

    #[test]
    fn from_str() {
        assert_eq!(
            Alias::from_string("application/x-foo application/foo").unwrap(),
            Alias::new(
                &"application/x-foo".parse().unwrap(),
                &"application/foo".parse().unwrap(),
            )
        );
    }

    #[test]
    fn extra_tokens_yield_error() {
        assert!(Alias::from_string("one/foo two/foo three/foo").is_none());
    }
}
