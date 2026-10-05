//! Direct interrogation of kete/manifestus overwrite visibility — written
//! after a false alarm (a swallowed identity-scope error masqueraded as
//! "rapid overwrites vanish"), to establish the engine's actual guarantees.
//! One test fn: the vault-dir override is process-global.

use std::sync::Arc;

fn fresh_store(tag: &str) -> (Arc<mahere_store::FlatStorage>, std::path::PathBuf) {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("kete-stress-{tag}-{nanos}"));
    std::fs::create_dir_all(&dir).unwrap();
    let store = mahere_store::open(Some(&dir.to_string_lossy())).expect("open");
    (store, dir)
}

#[test]
fn overwrite_visibility_gauntlet() {
    // 1. Rapid same-key overwrites: read == last, always.
    let (store, dir) = fresh_store("rapid");
    for i in 0u32..1000 {
        store.write_device("k", format!("value-{i}").as_bytes()).unwrap();
    }
    assert_eq!(
        store.read_device("k").unwrap().as_deref(),
        Some(b"value-999".as_slice()),
        "rapid overwrites: last write must win"
    );

    // 2. Interleaved big blob + small counter, the track-recorder shape.
    let blob = vec![0xA5u8; 16 * 1024];
    for i in 0u32..100 {
        store.write_device(&format!("blob|{i}"), &blob).unwrap();
        store.write_device("counter", &i.to_le_bytes()).unwrap();
    }
    assert_eq!(
        store.read_device("counter").unwrap().map(|v| u32::from_le_bytes(v.try_into().unwrap())),
        Some(99),
        "interleaved overwrites: counter must be the last value"
    );
    for i in (0u32..100).step_by(17) {
        assert_eq!(
            store.read_device(&format!("blob|{i}")).unwrap().map(|v| v.len()),
            Some(blob.len()),
            "blob {i} must read back whole"
        );
    }

    // 3. Delete, rewrite, read.
    store.delete_device("k").unwrap();
    assert_eq!(store.read_device("k").unwrap(), None, "deleted key must be absent");
    store.write_device("k", b"reborn").unwrap();
    assert_eq!(
        store.read_device("k").unwrap().as_deref(),
        Some(b"reborn".as_slice()),
        "rewrite after delete must be visible"
    );

    // 4. Concurrent writers: distinct keys from 4 threads, plus all four
    // hammering one shared key. Shared-key winner is nondeterministic, but
    // every read must return a COMPLETE value some thread wrote.
    let mut handles = Vec::new();
    for t in 0u32..4 {
        let s = store.clone();
        handles.push(std::thread::spawn(move || {
            for i in 0u32..200 {
                s.write_device(&format!("thread{t}|{i}"), &[t as u8; 64]).unwrap();
                s.write_device("shared", format!("t{t}-i{i}").as_bytes()).unwrap();
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    for t in 0u32..4 {
        for i in (0u32..200).step_by(37) {
            let v = store.read_device(&format!("thread{t}|{i}")).unwrap();
            assert_eq!(v.as_deref(), Some([t as u8; 64].as_slice()), "thread{t}|{i}");
        }
    }
    let shared = store.read_device("shared").unwrap().expect("shared key must exist");
    let text = String::from_utf8(shared).expect("shared value must be a complete utf8 write");
    assert!(
        text.starts_with('t') && text.contains("-i"),
        "shared value must be one thread's complete write, got {text:?}"
    );

    // 5. Overwritten values must survive a real reopen (drop every handle).
    store.write_device("persist", b"final-form").unwrap();
    drop(store);
    let store2 = mahere_store::open(Some(&dir.to_string_lossy())).expect("reopen");
    assert_eq!(
        store2.read_device("persist").unwrap().as_deref(),
        Some(b"final-form".as_slice()),
        "overwritten key must survive reopen"
    );
    assert_eq!(
        store2.read_device("counter").unwrap().map(|v| u32::from_le_bytes(v.try_into().unwrap())),
        Some(99),
        "interleaved counter must survive reopen"
    );
    assert_eq!(
        store2.read_device("k").unwrap().as_deref(),
        Some(b"reborn".as_slice()),
        "post-delete rewrite must survive reopen"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
