/*
    MicroHydraGPT Sandbox (TUI)
    A visual, interactive laboratory for observing evolutionary dynamics.
*/

use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::{Backend, CrosstermBackend},
    layout::{Constraint, Direction, Layout, Alignment},
    style::{Color, Modifier, Style},
    text::{Span, Line},
    widgets::{Block, Borders, Paragraph, Wrap},
    Terminal,
};
use microgpt_rust::{
    load_training_data, 
    Genome,
    heads::{Head, Weaver, Mirror, Spark, Origin}
};
use rand::prelude::*;
use rayon::prelude::*;
use std::collections::HashSet;
use std::error::Error;
use std::io;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

const INPUT_FILE: &str = "input.txt";

#[derive(Clone)]
struct HydraHead {
    // We use Arc<dyn Head> here to allow cloning the HydraHead struct easily for the UI thread
    head_impl: Arc<dyn Head>,
    population: Vec<Genome>,
    generation: usize,
    best_score: f64,
}

impl HydraHead {
    fn new(head_impl: Arc<dyn Head>) -> Self {
        HydraHead {
            head_impl,
            population: (0..8).map(|_| Genome::new_random()).collect(),
            generation: 0,
            best_score: 0.0,
        }
    }

    fn name(&self) -> &str {
        self.head_impl.name()
    }

    fn evolve(&mut self, training_data: &HashSet<String>) {
        let head_ref = self.head_impl.as_ref();
        self.population.par_iter_mut().for_each(|genome| {
            // Short training for TUI responsiveness (20 steps)
            genome.evaluate(training_data, head_ref, INPUT_FILE, 20);
        });
        self.population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        self.best_score = self.population[0].fitness;
        self.generation += 1;
    }

    fn breed(&mut self) {
        let elitism = 2;
        let mut new_pop = Vec::with_capacity(8);
        for i in 0..elitism { new_pop.push(self.population[i].clone()); }
        let mut rng = rand::thread_rng();
        while new_pop.len() < 8 {
            let parent = &self.population[rng.gen_range(0..elitism)];
            let mut child = parent.clone();
            child.mutate();
            new_pop.push(child);
        }
        self.population = new_pop;
    }
    
    fn inject_immigrants(&mut self, immigrants: Vec<Genome>) {
        let count = immigrants.len();
        let start = self.population.len() - count;
        for (i, immigrant) in immigrants.into_iter().enumerate() {
            if start + i < self.population.len() {
                let mut new_genome = immigrant;
                new_genome.fitness = 0.0;
                new_genome.names.clear();
                self.population[start + i] = new_genome;
            }
        }
    }
}

// --- Application State ---

struct App {
    heads: Vec<HydraHead>,
    training_data: Arc<HashSet<String>>,
    running: bool,
    cycle: usize,
    message: String,
}

impl App {
    fn new() -> Self {
        // Load data synchronously on startup
        if std::fs::metadata(INPUT_FILE).is_err() {
            let _ = std::process::Command::new("curl")
                .args(["-o", INPUT_FILE, "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"])
                .output();
        }
        let raw = load_training_data(INPUT_FILE);
        let training_data: HashSet<String> = raw.lines().map(|l| l.trim().to_lowercase()).collect();

        App {
            heads: vec![
                HydraHead::new(Arc::new(Weaver)),
                HydraHead::new(Arc::new(Mirror)),
                HydraHead::new(Arc::new(Spark)),
                HydraHead::new(Arc::new(Origin)),
            ],
            training_data: Arc::new(training_data),
            running: false,
            cycle: 0,
            message: "Press 'r' to start/pause evolution, 'q' to quit.".to_string(),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // App
    let app = Arc::new(Mutex::new(App::new()));
    
    // Background Evolution Thread
    let app_runner = app.clone();
    thread::spawn(move || {
        loop {
            // 1. Snapshot State (Lock briefly)
            let (should_run, heads, data) = {
                let app = app_runner.lock().unwrap();
                if app.running {
                    (true, Some(app.heads.clone()), Some(app.training_data.clone()))
                } else {
                    (false, None, None)
                }
            };

            // 2. Evolve (No Lock)
            if should_run {
                if let (Some(mut heads), Some(data)) = (heads, data) {
                    heads.par_iter_mut().for_each(|head| {
                        head.evolve(&data);
                        head.breed();
                    });

                    // 3. Update State (Lock briefly)
                    {
                        let mut app = app_runner.lock().unwrap();
                        app.heads = heads;
                        app.cycle += 1;
                        
                        // Cross-Pollination (Every 5 cycles)
                        if app.cycle % 5 == 0 {
                            app.message = format!("Cycle {}: Cross-Pollination!", app.cycle);
                            let mut pool = Vec::new();
                            for head in &app.heads {
                                pool.push(head.population[0].clone());
                            }
                            let immigrants = pool.clone();
                            for head in &mut app.heads {
                                head.inject_immigrants(immigrants.clone());
                            }
                        } else {
                            app.message = format!("Cycle {}: Evolving...", app.cycle);
                        }
                    }
                }
            } else {
                thread::sleep(Duration::from_millis(100));
            }
        }
    });

    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}

fn run_app<B: Backend>(terminal: &mut Terminal<B>, app_state: Arc<Mutex<App>>) -> io::Result<()> {
    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        {
            let app = app_state.lock().unwrap();
            terminal.draw(|f| ui(f, &app))?;
        }

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                let mut app = app_state.lock().unwrap();
                match key.code {
                    KeyCode::Char('q') => return Ok(()),
                    KeyCode::Char('r') => {
                        app.running = !app.running;
                        if app.running {
                            app.message = format!("Cycle {}: Resumed.", app.cycle);
                        } else {
                            app.message = format!("Cycle {}: Paused.", app.cycle);
                        }
                    },
                    _ => {}
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            last_tick = Instant::now();
        }
    }
}

fn ui(f: &mut ratatui::Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Title
            Constraint::Min(0),    // Heads
            Constraint::Length(3), // Status
        ].as_ref())
        .split(f.size());

    let title = Paragraph::new(Line::from(vec![
        Span::styled("MicroHydraGPT Sandbox", Style::default().add_modifier(Modifier::BOLD).fg(Color::Cyan)),
        Span::raw(" | "),
        Span::styled("Interactive Evolutionary Lab", Style::default().fg(Color::Gray)),
    ]))
    .alignment(Alignment::Center)
    .block(Block::default().borders(Borders::ALL));
    f.render_widget(title, chunks[0]);

    // Split middle into 4 quadrants for 4 heads
    let main_chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(chunks[1]);
    
    let top_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(main_chunks[0]);
        
    let bot_row = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(main_chunks[1]);

    let areas = vec![top_row[0], top_row[1], bot_row[0], bot_row[1]];

    for (i, head) in app.heads.iter().enumerate() {
        if i >= 4 { break; }
        
        let best = &head.population[0];
        let samples = best.names.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
        
        let color = match head.name() {
            "Weaver" => Color::Green,
            "Mirror" => Color::Magenta,
            "Spark" => Color::Yellow,
            "Origin" => Color::Blue,
            _ => Color::White,
        };

        let content = vec![
            Line::from(vec![Span::raw("Objective: "), Span::styled(head.name(), Style::default().fg(color))]),
            Line::from(vec![Span::raw(format!("Best Score: {:.4} (Gen {})", best.fitness, head.generation))]),
            Line::from(vec![Span::raw(format!("Genome: E:{} H:{} L:{} C:{} F:{}", best.n_emb, best.n_head, best.n_layer, best.n_ctx, best.n_ff_exp))]),
            Line::from(vec![Span::raw(format!("LR: {:.5}", best.lr))]),
            Line::from(vec![Span::raw("")]),
            Line::from(vec![Span::styled("Samples:", Style::default().add_modifier(Modifier::UNDERLINED))]),
            Line::from(vec![Span::raw(samples)]),
        ];

        let block = Paragraph::new(content)
            .block(Block::default().title(head.name()).borders(Borders::ALL).border_style(Style::default().fg(color)))
            .wrap(Wrap { trim: true });
        
        f.render_widget(block, areas[i]);
    }

    let status_text = format!("{} | Cycle: {}", app.message, app.cycle);
    let status = Paragraph::new(status_text)
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(status, chunks[2]);
}
