use std::{
    collections::HashMap,
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use mime::Mime;

pub fn read_comments_from_dir<P: AsRef<Path>>(dir: P) -> HashMap<Mime, String> {
    let dir = dir.as_ref();
    let file_name = dir.join("types");
    let mut res = HashMap::new();

    let f = match File::open(file_name) {
        Ok(v) => v,
        Err(_) => return res,
    };

    let file = BufReader::new(&f);
    for line in file.lines() {
        if line.is_err() {
            return res; // FIXME: return error instead
        }

        let line = line.unwrap();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        match line.parse() {
            Ok(v) => match read_comment_from_file(dir.join(line + ".xml")) {
                Some(c) => {
                    res.entry(v).or_insert(c);
                }
                None => {}
            },
            Err(_) => continue,
        }
    }

    res
}

pub fn read_comment_from_file<P: AsRef<Path>>(file: P) -> Option<String> {
    let f = match File::open(file) {
        Ok(v) => v,
        Err(_) => return None,
    };

    let file = BufReader::new(&f);
    for line in file.lines() {
        if line.is_err() {
            return None; // FIXME: return error instead
        }

        let line = line.unwrap();

        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        if let Some(line) = line.trim().strip_prefix("<comment>") {
            if let Some(line) = line.strip_suffix("</comment>") {
                return Some(line.to_owned());
            }
        }
    }

    None
}
