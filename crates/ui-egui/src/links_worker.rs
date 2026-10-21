//! Native background refreshes for the Links panel.
//!
//! Link status may touch local or network files. Keeping that work off the egui thread means an
//! unavailable share cannot stop the rest of the application from repainting or accepting input.

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};
    use std::{hash::Hash, hash::Hasher};

    use designcraft_compose::Cache;
    use designcraft_doc::Document;
    use designcraft_engine::cmd::LinkListChecks;
    use serde_json::Value;

    // Network metadata for a document with hundreds of links can itself take seconds. Document
    // edits change the key and refresh immediately; external-only changes are polled less often.
    const REFRESH_AFTER: Duration = Duration::from_secs(5 * 60);

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    pub(crate) struct Key {
        pub(crate) uid: u64,
        document: usize,
        path: u64,
    }

    impl Key {
        pub(crate) fn new(uid: u64, document: &Arc<Document>, path: Option<&str>) -> Self {
            let mut path_hash = std::collections::hash_map::DefaultHasher::new();
            path.hash(&mut path_hash);
            Self { uid, document: Arc::as_ptr(document) as usize, path: path_hash.finish() }
        }
    }

    pub(crate) struct Snapshot {
        pub(crate) key: Key,
        pub(crate) doc: Arc<Document>,
        pub(crate) cache: Arc<Cache>,
        pub(crate) document: Option<PathBuf>,
    }

    pub(crate) struct Rows {
        pub(crate) values: Arc<Vec<Value>>,
        pub(crate) checking: bool,
        pub(crate) unavailable: bool,
    }

    struct Job {
        token: u64,
        snapshot: Snapshot,
    }

    struct Done {
        token: u64,
        key: Key,
        values: Vec<Value>,
        complete: bool,
    }

    enum Poll {
        Done(Done),
        Empty,
        Disconnected,
    }

    type Runner = Box<dyn FnMut(&Snapshot, &mut LinkListChecks) -> Vec<Value> + Send>;

    struct Worker {
        tx: std::sync::mpsc::Sender<Job>,
        rx: std::sync::mpsc::Receiver<Done>,
    }

    impl Worker {
        fn spawn(ctx: egui::Context) -> Option<Self> {
            Self::spawn_with_preview(
                ctx,
                Some(Box::new(|snapshot, _| {
                    designcraft_engine::cmd::preview_document_links(&snapshot.doc, &snapshot.cache, snapshot.document.as_deref())
                })),
                Box::new(|snapshot, checks| {
                    designcraft_engine::cmd::list_document_links(
                        &snapshot.doc,
                        &snapshot.cache,
                        snapshot.document.as_deref(),
                        checks,
                    )
                }),
            )
        }

        #[cfg(test)]
        fn spawn_with(ctx: egui::Context, run: Runner) -> Option<Self> {
            Self::spawn_with_preview(ctx, None, run)
        }

        fn spawn_with_preview(ctx: egui::Context, mut preview: Option<Runner>, mut run: Runner) -> Option<Self> {
            let (tx, jobs) = std::sync::mpsc::channel::<Job>();
            let (done_tx, rx) = std::sync::mpsc::channel::<Done>();
            std::thread::Builder::new()
                .name("designcraft-links".into())
                .spawn(move || {
                    let mut checks = LinkListChecks::default();
                    while let Ok(job) = jobs.recv() {
                        if let Some(preview) = &mut preview {
                            let values = preview(&job.snapshot, &mut checks);
                            if done_tx.send(Done { token: job.token, key: job.snapshot.key, values, complete: false }).is_err() {
                                break;
                            }
                            ctx.request_repaint();
                        }
                        let values = run(&job.snapshot, &mut checks);
                        if done_tx.send(Done { token: job.token, key: job.snapshot.key, values, complete: true }).is_err() {
                            break;
                        }
                        ctx.request_repaint();
                    }
                })
                .ok()?;
            Some(Self { tx, rx })
        }

        fn submit(&self, job: Job) -> bool {
            self.tx.send(job).is_ok()
        }

        fn poll(&self) -> Poll {
            let mut latest = None;
            loop {
                match self.rx.try_recv() {
                    Ok(done) => latest = Some(done),
                    Err(std::sync::mpsc::TryRecvError::Empty) => return latest.map_or(Poll::Empty, Poll::Done),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => return latest.map_or(Poll::Disconnected, Poll::Done),
                }
            }
        }
    }

    struct Cached {
        key: Key,
        refreshed: Instant,
        values: Arc<Vec<Value>>,
    }

    #[derive(Default)]
    pub(crate) struct State {
        worker: Option<Worker>,
        worker_failed: bool,
        next_token: u64,
        in_flight: Option<(u64, Key)>,
        cached: Option<Cached>,
    }

    impl State {
        pub(crate) fn rows(&mut self, ctx: &egui::Context, snapshot: Snapshot) -> Rows {
            let current_key = snapshot.key;
            if self.worker.is_none() && !self.worker_failed {
                self.worker = Worker::spawn(ctx.clone());
                self.worker_failed = self.worker.is_none();
            }

            match self.worker.as_ref().map(Worker::poll) {
                Some(Poll::Done(done)) => {
                    if done.complete && self.in_flight.is_some_and(|(token, _)| token == done.token) {
                        self.in_flight = None;
                    }
                    if done.key == current_key {
                        self.cached = Some(Cached { key: done.key, refreshed: Instant::now(), values: Arc::new(done.values) });
                    }
                }
                Some(Poll::Disconnected) => {
                    self.worker = None;
                    self.worker_failed = true;
                    self.in_flight = None;
                }
                Some(Poll::Empty) | None => {}
            }

            let needs_refresh = self.cached.as_ref().is_none_or(|cached| cached.key != current_key || cached.refreshed.elapsed() >= REFRESH_AFTER);
            if needs_refresh && self.in_flight.is_none() && !self.worker_failed {
                self.next_token = self.next_token.wrapping_add(1);
                let token = self.next_token;
                let key = snapshot.key;
                if self.worker.as_ref().is_some_and(|worker| worker.submit(Job { token, snapshot })) {
                    self.in_flight = Some((token, key));
                } else {
                    self.worker = None;
                    self.worker_failed = true;
                }
            }

            let values = self
                .cached
                .as_ref()
                .filter(|cached| cached.key.uid == current_key.uid)
                .map_or_else(|| Arc::new(Vec::new()), |cached| Arc::clone(&cached.values));
            let checking = self.in_flight.is_some();
            if !checking && !self.worker_failed
                && let Some(cached) = self.cached.as_ref().filter(|cached| cached.key == current_key)
            {
                ctx.request_repaint_after(REFRESH_AFTER.saturating_sub(cached.refreshed.elapsed()));
            }
            Rows { values, checking, unavailable: self.worker_failed }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn snapshot(key: Key) -> Snapshot {
            Snapshot {
                key,
                doc: Arc::new(Document::new(&Default::default())),
                cache: Arc::new(Cache::new()),
                document: None,
            }
        }

        fn key(uid: u64, document: usize) -> Key {
            Key { uid, document, path: 0 }
        }

        #[test]
        fn a_blocked_link_check_does_not_block_the_ui_caller() {
            let ctx = egui::Context::default();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let watchdog = release_tx.clone();
            let _watchdog = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(1));
                let _ = watchdog.send(());
            });
            let worker = Worker::spawn_with(
                ctx.clone(),
                Box::new(move |_, _| {
                    let _ = started_tx.send(());
                    let _ = release_rx.recv();
                    Vec::new()
                }),
            )
            .unwrap();
            let mut state = State { worker: Some(worker), ..Default::default() };

            let before = Instant::now();
            let rows = state.rows(&ctx, snapshot(key(1, 1)));
            let elapsed = before.elapsed();
            let started = started_rx.recv_timeout(Duration::from_secs(1)).is_ok();
            let _ = release_tx.send(());

            assert!(rows.checking);
            assert!(started, "the filesystem check starts on the worker");
            assert!(elapsed < Duration::from_millis(500), "the UI call waited {elapsed:?}");
        }

        #[test]
        fn cached_rows_stay_visible_while_a_refresh_is_blocked() {
            let ctx = egui::Context::default();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let watchdog = release_tx.clone();
            let _watchdog = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(1));
                let _ = watchdog.send(());
            });
            let worker = Worker::spawn_with(
                ctx.clone(),
                Box::new(move |_, _| {
                    let _ = started_tx.send(());
                    let _ = release_rx.recv();
                    Vec::new()
                }),
            )
            .unwrap();
            let key = key(1, 1);
            let values = Arc::new(vec![serde_json::json!({"name": "cached"})]);
            let mut state = State {
                worker: Some(worker),
                cached: Some(Cached { key, refreshed: Instant::now() - REFRESH_AFTER, values }),
                ..Default::default()
            };

            let rows = state.rows(&ctx, snapshot(key));
            let started = started_rx.recv_timeout(Duration::from_secs(1)).is_ok();
            let _ = release_tx.send(());

            assert!(rows.checking);
            assert!(started, "the refresh starts on the worker");
            assert_eq!(rows.values.first().and_then(|value| value["name"].as_str()), Some("cached"));
        }

        #[test]
        fn completed_rows_stay_visible_when_document_content_changes() {
            let ctx = egui::Context::default();
            let (started_tx, started_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let watchdog = release_tx.clone();
            let _watchdog = std::thread::spawn(move || {
                std::thread::sleep(Duration::from_secs(1));
                let _ = watchdog.send(());
            });
            let worker = Worker::spawn_with(
                ctx.clone(),
                Box::new(move |_, _| {
                    let _ = started_tx.send(());
                    let _ = release_rx.recv();
                    Vec::new()
                }),
            )
            .unwrap();
            let values = Arc::new(vec![serde_json::json!({"name": "cached"})]);
            let mut state = State {
                worker: Some(worker),
                cached: Some(Cached { key: key(1, 1), refreshed: Instant::now(), values }),
                ..Default::default()
            };

            let rows = state.rows(&ctx, snapshot(key(1, 2)));
            let started = started_rx.recv_timeout(Duration::from_secs(1)).is_ok();
            let _ = release_tx.send(());

            assert!(rows.checking);
            assert!(started, "changed document content starts a refresh");
            assert_eq!(rows.values.first().and_then(|value| value["name"].as_str()), Some("cached"));
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) use native::{Key, Snapshot, State};

#[cfg(target_arch = "wasm32")]
#[derive(Default)]
pub(crate) struct State;
