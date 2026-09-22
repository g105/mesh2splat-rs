//! The viewport shares its rect with the transform gizmo, and
//! `transform-gizmo-egui` registers a 1x1 `click_and_drag` widget under the
//! cursor on every frame. Being added last it sits on top, so once the gizmo is
//! on screen the viewport's own `Response` stops seeing drags — which is why
//! the camera reads the raw pointer state instead. These tests pin both halves
//! of that down.

#![cfg(feature = "gui")]

use egui::{Event, Modifiers, PointerButton, Pos2, RawInput, Rect, Sense, Vec2};
use transform_gizmo_egui::math::{DMat4, DQuat, DVec3, Transform};
use transform_gizmo_egui::{Gizmo, GizmoConfig, GizmoExt, GizmoMode};

/// What a left-button drag across the viewport looks like to the app.
struct Drag {
    /// Delta the viewport's own `Response` reported.
    response: Vec2,
    /// Delta from the raw pointer state, with the button held.
    pointer: Vec2,
}

/// Hovers, presses the left button in the middle of the viewport with
/// `modifiers`, then moves the pointer. `with_gizmo` also runs the transform
/// gizmo over the same rect, as the app does.
fn drag_delta(modifiers: Modifiers, with_gizmo: bool) -> Drag {
    let ctx = egui::Context::default();
    let mut gizmo = Gizmo::default();
    let center = Pos2::new(400.0, 300.0);
    let screen = Rect::from_min_size(Pos2::ZERO, Vec2::new(800.0, 600.0));

    let mut frame = |events: Vec<Event>, pos: Pos2| -> Drag {
        let input = RawInput {
            screen_rect: Some(screen),
            events,
            modifiers,
            ..Default::default()
        };
        let mut delta = Drag {
            response: Vec2::ZERO,
            pointer: Vec2::ZERO,
        };
        ctx.run(input, |ctx| {
            egui::CentralPanel::default().show(ctx, |ui| {
                let rect = ui.max_rect();
                let response = ui.allocate_rect(rect, Sense::click_and_drag());
                if response.dragged_by(PointerButton::Primary) {
                    delta.response = response.drag_delta();
                }
                ui.input(|i| {
                    if i.pointer.button_down(PointerButton::Primary) {
                        delta.pointer = i.pointer.delta();
                    }
                });
                if with_gizmo {
                    // A gizmo at the origin, centred on screen and under the pointer.
                    gizmo.update_config(GizmoConfig {
                        view_matrix: DMat4::look_at_rh(
                            DVec3::new(0.0, 0.0, 5.0),
                            DVec3::ZERO,
                            DVec3::Y,
                        )
                        .into(),
                        projection_matrix: DMat4::perspective_rh(
                            45f64.to_radians(),
                            rect.width() as f64 / rect.height() as f64,
                            0.01,
                            100.0,
                        )
                        .into(),
                        viewport: rect,
                        modes: GizmoMode::all_translate(),
                        ..Default::default()
                    });
                    let t = Transform::from_scale_rotation_translation(
                        DVec3::ONE,
                        DQuat::IDENTITY,
                        DVec3::ZERO,
                    );
                    let _ = gizmo.interact(ui, &[t]);
                }
            });
        });
        let _ = pos;
        delta
    };

    // Hover first, as in a running app: the gizmo puts its interaction widget
    // wherever the cursor was last frame.
    frame(vec![Event::PointerMoved(center)], center);
    frame(vec![], center);
    frame(
        vec![
            Event::PointerMoved(center),
            Event::PointerButton {
                pos: center,
                button: PointerButton::Primary,
                pressed: true,
                modifiers,
            },
        ],
        center,
    );
    let moved = center + Vec2::new(40.0, 10.0);
    frame(vec![Event::PointerMoved(moved)], moved)
}

/// Without a gizmo the response sees the drag, and so does the pointer state.
#[test]
fn left_drag_without_a_gizmo() {
    let d = drag_delta(Modifiers::default(), false);
    assert!(d.response.x > 1.0, "response: {:?}", d.response);
    assert!(d.pointer.x > 1.0, "pointer: {:?}", d.pointer);
}

/// With a gizmo on screen the response no longer sees it: this is the bug that
/// stopped Alt + left drag from tumbling while the gizmo was displayed.
#[test]
fn a_gizmo_takes_the_drag_from_the_response() {
    let d = drag_delta(Modifiers::ALT, true);
    assert_eq!(d.response, Vec2::ZERO, "gizmo no longer steals the drag");
}

/// The raw pointer state still reports it, which is what the camera uses.
#[test]
fn the_pointer_state_survives_a_gizmo() {
    let d = drag_delta(Modifiers::ALT, true);
    assert!(d.pointer.x > 1.0, "pointer: {:?}", d.pointer);
}
