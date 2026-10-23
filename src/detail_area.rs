// SPDX-License-Identifier: GPL-3.0-or-later

//! Input surface for the detail-view preview.
//!
//! A thin wrapper modeled on iced's `MouseArea` that publishes cursor
//! positions **relative to the widget center** as well as press/release and
//! wheel events. iced's plain `MouseArea.on_move` delivers top-left-relative
//! positions; the detail view's zoom/pan math compares the cursor against the
//! pan offset (which is also widget-center-relative), so mixing frames would
//! shift every zoom anchor by the unknown widget center. Everything else
//! delegates to the wrapped content, so it behaves like a plain container to
//! the renderer.

use cosmic::iced::core::layout;
use cosmic::iced::core::mouse;
use cosmic::iced::core::overlay;
use cosmic::iced::core::renderer;
use cosmic::iced::core::touch;
use cosmic::iced::core::widget::{Operation, Tree, tree};
use cosmic::iced::core::{
    Clipboard, Element, Event, Layout, Length, Point, Rectangle, Shell, Size, Vector, Widget,
};

/// The cursor position relative to the widget center, given `position` already
/// relative to the widget's top-left (`cursor.position_in(bounds)`) and the
/// widget `bounds`. The half-size comes from the bounds' size, not
/// `bounds.center()` (which is in absolute coordinates) — mixing the two frames
/// offset every reported point by the widget's absolute origin.
#[must_use]
fn center_relative(bounds: Rectangle, position: Point) -> Point {
    Point::new(
        position.x - bounds.width / 2.0,
        position.y - bounds.height / 2.0,
    )
}

/// Emits zoom/pan messages on mouse events over the detail preview.
pub struct DetailArea<'a, Message, Theme = cosmic::Theme, Renderer = cosmic::Renderer> {
    content: Element<'a, Message, Theme, Renderer>,
    /// Message to emit when the left button is pressed over the area.
    on_press: Option<Message>,
    /// Message to emit when the left button is released (or the cursor
    /// leaves while pressed) — ends a grab-pan drag.
    on_release: Option<Message>,
    /// Called with the cursor position relative to the widget **center**.
    on_move: Option<Box<dyn Fn(Point) -> Message + 'a>>,
    /// Called on wheel scroll with the scroll delta.
    on_scroll: Option<Box<dyn Fn(mouse::ScrollDelta) -> Message + 'a>>,
    /// Called with the widget's logical size whenever its layout bounds change.
    on_resize: Option<Box<dyn Fn(Size) -> Message + 'a>>,
    /// [`mouse::Interaction`] to use when hovering the area.
    interaction: Option<mouse::Interaction>,
}

/// Local state of the [`DetailArea`].
#[derive(Default)]
struct State {
    was_over: bool,
    pressed: bool,
    /// Last reported layout size, so `on_resize` only fires on actual changes
    /// (publishing on every event would spin a message loop).
    last_size: Size,
}

impl<'a, Message, Theme, Renderer> DetailArea<'a, Message, Theme, Renderer> {
    /// Creates a [`DetailArea`] with the given content.
    pub fn new(content: impl Into<Element<'a, Message, Theme, Renderer>>) -> Self {
        Self {
            content: content.into(),
            on_press: None,
            on_release: None,
            on_move: None,
            on_scroll: None,
            on_resize: None,
            interaction: None,
        }
    }

    /// The message to emit on a left button press over the area.
    #[must_use]
    pub fn on_press(mut self, message: Message) -> Self {
        self.on_press = Some(message);
        self
    }

    /// The message to emit on a left button release (or when the cursor
    /// leaves the area while the button is held).
    #[must_use]
    pub fn on_release(mut self, message: Message) -> Self {
        self.on_release = Some(message);
        self
    }

    /// The message to emit when the mouse moves over the area; receives the
    /// cursor position relative to the widget center in logical points.
    #[must_use]
    pub fn on_move(mut self, on_move: impl Fn(Point) -> Message + 'a) -> Self {
        self.on_move = Some(Box::new(on_move));
        self
    }

    /// The message to emit when the wheel is scrolled over the area.
    #[must_use]
    pub fn on_scroll(mut self, on_scroll: impl Fn(mouse::ScrollDelta) -> Message + 'a) -> Self {
        self.on_scroll = Some(Box::new(on_scroll));
        self
    }

    /// The message to emit with the widget's logical size whenever its layout
    /// bounds change (window resize, drawer open/close, …). The detail view
    /// uses it to recompute the 1:1 ("100%") zoom cap, which depends on the
    /// preview area's size.
    #[must_use]
    pub fn on_resize(mut self, on_resize: impl Fn(Size) -> Message + 'a) -> Self {
        self.on_resize = Some(Box::new(on_resize));
        self
    }

    /// The [`mouse::Interaction`] to use when hovering the area.
    #[must_use]
    pub fn interaction(mut self, interaction: mouse::Interaction) -> Self {
        self.interaction = Some(interaction);
        self
    }
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer>
    for DetailArea<'_, Message, Theme, Renderer>
where
    Renderer: renderer::Renderer,
    Message: Clone,
{
    fn tag(&self) -> tree::Tag {
        tree::Tag::of::<State>()
    }

    fn state(&self) -> tree::State {
        tree::State::new(State::default())
    }

    fn children(&self) -> Vec<Tree> {
        vec![Tree::new(&self.content)]
    }

    fn diff(&mut self, tree: &mut Tree) {
        tree.diff_children(std::slice::from_mut(&mut self.content));
    }

    fn size(&self) -> Size<Length> {
        self.content.as_widget().size()
    }

    fn layout(
        &mut self,
        tree: &mut Tree,
        renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        self.content
            .as_widget_mut()
            .layout(&mut tree.children[0], renderer, limits)
    }

    fn operate(
        &mut self,
        tree: &mut Tree,
        layout: Layout<'_>,
        renderer: &Renderer,
        operation: &mut dyn Operation,
    ) {
        self.content
            .as_widget_mut()
            .operate(&mut tree.children[0], layout, renderer, operation);
    }

    fn update(
        &mut self,
        tree: &mut Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        renderer: &Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        viewport: &Rectangle,
    ) {
        self.content.as_widget_mut().update(
            &mut tree.children[0],
            event,
            layout,
            cursor,
            renderer,
            clipboard,
            shell,
            viewport,
        );

        if shell.is_event_captured() {
            return;
        }

        let state: &mut State = tree.state.downcast_mut();
        let bounds = layout.bounds();
        let over = cursor.is_over(bounds);

        // Report layout size changes so the app can recompute the 1:1 zoom cap
        // (depends on the preview area's size). Guarded on `last_size` so it
        // fires once per actual change, not on every event.
        let size = bounds.size();
        if size != state.last_size {
            state.last_size = size;
            if let Some(on_resize) = self.on_resize.as_ref() {
                shell.publish(on_resize(size));
            }
        }

        match event {
            Event::Mouse(mouse::Event::CursorMoved { .. })
            | Event::Touch(touch::Event::FingerMoved { .. }) => {
                if over {
                    if let Some(on_move) = self.on_move.as_ref()
                        && let Some(position) = cursor.position_in(bounds)
                    {
                        shell.publish(on_move(center_relative(bounds, position)));
                    }
                } else if state.was_over && state.pressed {
                    // The cursor left while the button was held: end the drag
                    // so grab-pan never sticks (iced's MouseArea does not
                    // deliver a release outside its bounds).
                    state.pressed = false;
                    if let Some(on_release) = self.on_release.as_ref() {
                        shell.publish(on_release.clone());
                    }
                }
                state.was_over = over;
            }

            Event::Mouse(mouse::Event::CursorLeft)
            | Event::Touch(touch::Event::FingerLost { .. }) => {
                if state.pressed {
                    state.pressed = false;
                    if let Some(on_release) = self.on_release.as_ref() {
                        shell.publish(on_release.clone());
                    }
                }
                state.was_over = false;
            }

            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left))
            | Event::Touch(touch::Event::FingerPressed { .. })
                if over =>
            {
                state.pressed = true;
                if let Some(on_press) = self.on_press.as_ref() {
                    shell.publish(on_press.clone());
                }
                shell.capture_event();
            }

            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left))
            | Event::Touch(touch::Event::FingerLifted { .. })
                if state.pressed =>
            {
                state.pressed = false;
                if let Some(on_release) = self.on_release.as_ref() {
                    shell.publish(on_release.clone());
                }
                shell.capture_event();
            }

            Event::Mouse(mouse::Event::WheelScrolled { delta }) if over => {
                if let Some(on_scroll) = self.on_scroll.as_ref() {
                    shell.publish(on_scroll(*delta));
                    shell.capture_event();
                }
            }

            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        tree: &Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
        renderer: &Renderer,
    ) -> mouse::Interaction {
        let content_interaction = self.content.as_widget().mouse_interaction(
            &tree.children[0],
            layout,
            cursor,
            viewport,
            renderer,
        );

        match (self.interaction, content_interaction) {
            (Some(interaction), mouse::Interaction::None) if cursor.is_over(layout.bounds()) => {
                interaction
            }
            _ => content_interaction,
        }
    }

    fn draw(
        &self,
        tree: &Tree,
        renderer: &mut Renderer,
        theme: &Theme,
        renderer_style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        viewport: &Rectangle,
    ) {
        self.content.as_widget().draw(
            &tree.children[0],
            renderer,
            theme,
            renderer_style,
            layout,
            cursor,
            viewport,
        );
    }

    fn overlay<'b>(
        &'b mut self,
        tree: &'b mut Tree,
        layout: Layout<'b>,
        renderer: &Renderer,
        viewport: &Rectangle,
        translation: Vector,
    ) -> Option<overlay::Element<'b, Message, Theme, Renderer>> {
        self.content.as_widget_mut().overlay(
            &mut tree.children[0],
            layout,
            renderer,
            viewport,
            translation,
        )
    }
}

impl<'a, Message, Theme, Renderer> From<DetailArea<'a, Message, Theme, Renderer>>
    for Element<'a, Message, Theme, Renderer>
where
    Message: 'a + Clone,
    Theme: 'a,
    Renderer: 'a + renderer::Renderer,
{
    fn from(area: DetailArea<'a, Message, Theme, Renderer>) -> Self {
        Element::new(area)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_relative_uses_the_bounds_size_not_their_absolute_origin() {
        // A widget below the header (absolute origin y = 48): the reported point
        // must be independent of the widget's absolute position.
        let bounds = Rectangle {
            x: 0.0,
            y: 48.0,
            width: 400.0,
            height: 300.0,
        };
        assert_eq!(
            center_relative(bounds, Point::new(0.0, 0.0)),
            Point::new(-200.0, -150.0)
        );
        assert_eq!(
            center_relative(bounds, Point::new(200.0, 150.0)),
            Point::new(0.0, 0.0)
        );
        assert_eq!(
            center_relative(bounds, Point::new(400.0, 300.0)),
            Point::new(200.0, 150.0)
        );
    }
}
