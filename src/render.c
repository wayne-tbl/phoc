/*
 * Copyright (C) 2021 Purism SPC
 * Copyright (C) 2023-2024 The Phosh Developers
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 *
 * Authors: The wlroots authors
 *          Sebastian Krzyszkowiak
 *          Guido Günther <agx@sigxcpu.org>
 */

#define G_LOG_DOMAIN "phoc-render"

#include "phoc-config.h"
#include "bling.h"
#include "layer-shell.h"
#include "seat.h"
#include "server.h"
#include "render.h"
#include "render-private.h"
#include "xwayland-surface.h"
#include "utils.h"

#define _POSIX_C_SOURCE 200809L
#include <assert.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>
#include <wlr/backend.h>
#include <wlr/config.h>
#include <wlr/render/drm_format_set.h>
#include <wlr/render/wlr_renderer.h>
#include <wlr/render/gles2.h>
#include <wlr/render/android.h>
#include <wlr/render/egl.h>
#include <wlr/types/wlr_compositor.h>
#include <wlr/types/wlr_matrix.h>
#include <wlr/types/wlr_buffer.h>
#include <wlr/types/wlr_linux_dmabuf_v1.h>
#include <wlr/util/region.h>
#include <wlr/render/allocator.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>

#define TOUCH_POINT_SIZE 20
#define TOUCH_POINT_BORDER 0.1

#define COLOR_BLACK                ((struct wlr_render_color){0.0f, 0.0f, 0.0f, 1.0f})
#define COLOR_TRANSPARENT          {0.0f, 0.0f, 0.0f, 0.0f}
#define COLOR_TRANSPARENT_WHITE    ((struct wlr_render_color){0.5f, 0.5f, 0.5f, 0.5f})
#define COLOR_TRANSPARENT_YELLOW   ((struct wlr_render_color){0.5f, 0.5f, 0.0f, 0.5f})
#define COLOR_TRANSPARENT_MAGENTA  ((struct wlr_render_color){0.5f, 0.0f, 0.5f, 0.5f})


/**
 * PhocRenderer:
 *
 * The renderer
 */

enum {
  RENDER_END,
  N_SIGNALS
};
static guint signals[N_SIGNALS] = { 0 };

enum {
  PROP_0,
  PROP_WLR_BACKEND,
  PROP_LAST_PROP
};
static GParamSpec *props[PROP_LAST_PROP];

/* Levels in the blur pyramid, level 0 being the quarter resolution base. Each
 * further level halves the resolution again and roughly doubles the reach. */
#define PHOC_BLUR_MAX_LEVELS 5

/* Blur resources are plain GL textures and FBOs on purpose: the android
 * renderer used on Halium devices has neither dmabuf import nor a native
 * window for offscreen buffers, so wlr_allocator buffers cannot be bound as
 * render targets there (see the "Do not use wlr_allocator on android" path in
 * phoc_renderer_render_view_to_buffer). */
typedef struct _PhocBlurState {
  int    width, height;    /* Output buffer size this state was built for */
  int    bw, bh;           /* Working resolution of the blur, i.e. level 0 */
  GLuint capture;          /* Full res copy of the output's framebuffer */
  /* The pyramid. Level i is half the size of level i - 1; the down pass walks
   * it outwards and the up pass walks it back, so no ping-pong is needed. */
  GLuint tex[PHOC_BLUR_MAX_LEVELS];
  GLuint fbo[PHOC_BLUR_MAX_LEVELS];
  int    w[PHOC_BLUR_MAX_LEVELS], h[PHOC_BLUR_MAX_LEVELS];
} PhocBlurState;


/* One compiled blur shader and the locations it was linked with */
typedef struct _PhocBlurProg {
  GLuint prog;
  GLint  attr_pos, attr_uv;
  GLint  uni_tex, uni_halfpixel, uni_srect;
} PhocBlurProg;


struct _PhocRenderer {
  GObject               parent;

  struct wlr_backend   *wlr_backend;
  struct wlr_renderer  *wlr_renderer;
  struct wlr_allocator *wlr_allocator;

  /* Whether blur is wanted at all, from the `blur` gsetting */
  gboolean              blur_enabled;
  /* Blur state per output */
  GHashTable           *blur_states;
  /* The two halves of the dual filter, plus the final composite which is the
   * up shader with a zero offset */
  PhocBlurProg          blur_down, blur_up;
  gboolean              blur_prog_failed;
  gboolean              blur_state_failed;
  /* Buffers and textures to keep alive until the frame was submitted */
  GSList               *frame_textures;
  GSList               *frame_buffers;
};

static void phoc_renderer_initable_iface_init (GInitableIface *iface);
static void blur_draw_quad (const PhocBlurProg *prog,
                            GLuint              texture,
                            const float        *pos,
                            const float        *uv,
                            float               half_x,
                            float               half_y,
                            const float        *srect);

G_DEFINE_TYPE_WITH_CODE (PhocRenderer, phoc_renderer, G_TYPE_OBJECT,
                         G_IMPLEMENT_INTERFACE (G_TYPE_INITABLE, phoc_renderer_initable_iface_init));


struct view_render_data {
  PhocView *view;
  int width;
  int height;
};

struct touch_point_data {
  int id;
  double x;
  double y;
};


static void
wlr_box_from_pixman_box32 (struct wlr_box *dest, const pixman_box32_t box)
{
  *dest = (struct wlr_box){
    .x = box.x1,
    .y = box.y1,
    .width = box.x2 - box.x1,
    .height = box.y2 - box.y1,
  };
}


static void
phoc_renderer_set_property (GObject      *object,
                            guint         property_id,
                            const GValue *value,
                            GParamSpec   *pspec)
{
  PhocRenderer *self = PHOC_RENDERER (object);

  switch (property_id) {
  case PROP_WLR_BACKEND:
    self->wlr_backend = g_value_get_pointer (value);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
    break;
  }
}


static void
phoc_renderer_get_property (GObject    *object,
                            guint       property_id,
                            GValue     *value,
                            GParamSpec *pspec)
{
  PhocRenderer *self = PHOC_RENDERER (object);

  switch (property_id) {
  case PROP_WLR_BACKEND:
    g_value_set_pointer (value, self->wlr_backend);
    break;
  default:
    G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
    break;
  }
}


static void
render_texture (PhocOutput               *output,
                struct wlr_texture       *texture,
                const struct wlr_fbox    *_src_box,
                const struct wlr_box     *dst_box,
                const struct wlr_box     *clip_box,
                enum wl_output_transform  surface_transform,
                float                     alpha,
                PhocRenderContext        *ctx)
{
  pixman_region32_t damage;
  struct wlr_box proj_box = *dst_box;
  struct wlr_fbox src_box = {0};
  enum wl_output_transform transform;

  if (!phoc_utils_is_damaged (&proj_box, ctx->damage, clip_box, &damage))
    goto buffer_damage_finish;

  if (_src_box)
    src_box = *_src_box;

  phoc_output_transform_box (output, &proj_box);
  phoc_output_transform_damage (output, &damage);
  transform = wlr_output_transform_compose (surface_transform, output->wlr_output->transform);

  wlr_render_pass_add_texture (ctx->render_pass, &(struct wlr_render_texture_options) {
      .texture = texture,
      .src_box = src_box,
      .dst_box = proj_box,
      .transform = transform,
      .alpha = &alpha,
      .clip = &damage,
      .filter_mode = phoc_output_get_texture_filter_mode (ctx->output),
    });

 buffer_damage_finish:
  pixman_region32_fini (&damage);
}

static void
collect_touch_points (PhocOutput *output, struct wlr_surface *surface, struct wlr_box box, float scale)
{
  PhocInput *input = phoc_server_get_input (phoc_server_get_default ());
  PhocServer *server = phoc_server_get_default ();
  if (G_LIKELY (!(phoc_server_check_debug_flags (server, PHOC_SERVER_DEBUG_FLAG_TOUCH_POINTS))))
    return;

  for (GSList *elem = phoc_input_get_seats (input); elem; elem = elem->next) {
    PhocSeat *seat = PHOC_SEAT (elem->data);
    struct wlr_touch_point *point;

    g_assert (PHOC_IS_SEAT (seat));

    wl_list_for_each(point, &seat->seat->touch_state.touch_points, link) {
      struct touch_point_data *touch_point;

      if (point->surface != surface)
        continue;

      touch_point = g_new (struct touch_point_data, 1);
      touch_point->id = point->touch_id;
      touch_point->x = box.x + point->sx * output->wlr_output->scale * scale;
      touch_point->y = box.y + point->sy * output->wlr_output->scale * scale;
      output->debug_touch_points = g_list_append (output->debug_touch_points, touch_point);
    }
  }
}


static void
render_surface_iterator (PhocOutput         *output,
                         struct wlr_surface *surface,
                         struct wlr_box     *box,
                         float               scale,
                         void               *data)
{
  PhocRenderContext *ctx = data;
  struct wlr_output *wlr_output = output->wlr_output;
  float alpha = ctx->alpha;

  struct wlr_texture *texture = wlr_surface_get_texture (surface);
  if (!texture)
    return;

  /* Anything drawn invalidates a blur capture taken before it */
  ctx->blur_stale = TRUE;

  struct wlr_fbox src_box;
  wlr_surface_get_buffer_source_box (surface, &src_box);

  struct wlr_box dst_box = *box;
  struct wlr_box clip_box = *box;

  phoc_utils_scale_box (&dst_box, scale);
  phoc_utils_scale_box (&dst_box, wlr_output->scale);
  phoc_utils_scale_box (&clip_box, scale);
  phoc_utils_scale_box (&clip_box, wlr_output->scale);

  render_texture (output, texture, &src_box, &dst_box, &clip_box, surface->current.transform, alpha, ctx);

  wlr_presentation_surface_scanned_out_on_output (output->desktop->presentation,
                                                  surface,
                                                  wlr_output);

  collect_touch_points(output, surface, dst_box, scale);
}


static void
render_blings (PhocOutput *output, PhocView *view, PhocRenderContext *ctx)
{
  GSList *blings;

  if (!phoc_view_is_mapped (view))
    return;

  blings = phoc_view_get_blings (view);
  if (!blings)
    return;

  for (GSList *l = blings; l; l = l->next) {
    PhocBling *bling = PHOC_BLING (l->data);

    phoc_bling_render (bling, ctx);
  }
}


static void     render_blur_backdrop_box  (struct wlr_box geo, PhocRenderContext *ctx);
static gboolean view_wants_blur           (PhocView *view);
static gboolean phoc_renderer_blur_enabled (PhocRenderer *self);
static void     phoc_renderer_capture_blur (PhocRenderer      *self,
                                            PhocOutput        *output,
                                            PhocRenderContext *ctx);

static void
render_view (PhocOutput *output, PhocView *view, PhocRenderContext *ctx)
{
  // Do not render views fullscreened on other outputs
  if (phoc_view_is_fullscreen (view) && phoc_view_get_fullscreen_output (view) != output)
    return;

  ctx->alpha = phoc_view_get_alpha (view);

  if (!phoc_view_is_fullscreen (view))
    render_blings (output, view, ctx);

  /* Frost whatever is behind a translucent window, the same way a blur enabled
   * layer surface gets it. Toplevels cannot ask for this over the layer shell
   * effects protocol, so the opaque region is what decides. */
  if (phoc_renderer_blur_enabled (ctx->renderer) &&
      phoc_output_get_blur_radius (output) > 0 &&
      view_wants_blur (view)) {
    struct wlr_box geo;

    phoc_view_get_box (view, &geo);
    geo.x -= output->lx;
    geo.y -= output->ly;

    if (ctx->blur_texture == 0 || ctx->blur_stale)
      phoc_renderer_capture_blur (ctx->renderer, output, ctx);

    if (ctx->blur_texture)
      render_blur_backdrop_box (geo, ctx);
  }

  phoc_output_view_for_each_surface (output, view, render_surface_iterator, ctx);
}



/*
 * Draw the blurred backdrop behind a blur enabled layer surface. The
 * backdrop texture covers the whole output in buffer coordinates so
 * we sample the region matching the surface's geometry.
 */
static void
render_blur_backdrop_box (struct wlr_box geo, PhocRenderContext *ctx)
{
  PhocOutput *output = ctx->output;
  struct wlr_output *wlr_output = output->wlr_output;
  PhocRenderer *self = ctx->renderer;
  struct wlr_box dst;
  float pos[8], uv[8];
  float x0, x1, y0, y1;
  int ow = wlr_output->width, oh = wlr_output->height;
  GLint prev_abuf = 0, prev_unit = GL_TEXTURE0;
  GLboolean prev_blend, prev_scissor;

  if (self == NULL || ow <= 0 || oh <= 0)
    return;

  dst = geo;
  phoc_utils_scale_box (&dst, wlr_output->scale);
  phoc_output_transform_box (output, &dst);

  /* A surface may extend past the output, keep the quad on screen */
  if (dst.x < 0) {
    dst.width += dst.x;
    dst.x = 0;
  }
  if (dst.y < 0) {
    dst.height += dst.y;
    dst.y = 0;
  }
  dst.width = MIN (dst.width, ow - dst.x);
  dst.height = MIN (dst.height, oh - dst.y);
  if (dst.width <= 0 || dst.height <= 0)
    return;

  /* Buffer coordinates are top left based, GL's are bottom left based */
  x0 = 2.0f * dst.x / ow - 1.0f;
  x1 = 2.0f * (dst.x + dst.width) / ow - 1.0f;
  y0 = 1.0f - 2.0f * (dst.y + dst.height) / oh;
  y1 = 1.0f - 2.0f * dst.y / oh;

  pos[0] = x0; pos[1] = y0;
  pos[2] = x1; pos[3] = y0;
  pos[4] = x0; pos[5] = y1;
  pos[6] = x1; pos[7] = y1;

  /* Sample exactly the pixels the surface covers, so the backdrop lines up
   * with what is behind it rather than sliding around as the surface moves */
  for (int i = 0; i < 8; i++)
    uv[i] = pos[i] * 0.5f + 0.5f;

  glGetIntegerv (GL_ARRAY_BUFFER_BINDING, &prev_abuf);
  glGetIntegerv (GL_ACTIVE_TEXTURE, &prev_unit);
  prev_blend = glIsEnabled (GL_BLEND);
  prev_scissor = glIsEnabled (GL_SCISSOR_TEST);

  /* The backdrop replaces what is underneath, the surface itself blends over
   * it afterwards. The whole output is damaged whenever blur is active, so no
   * scissoring is needed here. */
  glBindBuffer (GL_ARRAY_BUFFER, 0);
  glDisable (GL_BLEND);
  glDisable (GL_SCISSOR_TEST);

  blur_draw_quad (&self->blur_up, ctx->blur_texture, pos, uv, 0.0f, 0.0f, NULL);

  glBindBuffer (GL_ARRAY_BUFFER, prev_abuf);
  if (prev_blend)
    glEnable (GL_BLEND);
  if (prev_scissor)
    glEnable (GL_SCISSOR_TEST);
  glUseProgram (0);
  glBindTexture (GL_TEXTURE_2D, 0);
  glActiveTexture (prev_unit);
}


static void
render_blur_backdrop (PhocLayerSurface *layer_surface, PhocRenderContext *ctx)
{
  render_blur_backdrop_box (layer_surface->geo, ctx);
}


/*
 * Whether there is any point blurring behind this window.
 *
 * wlroots sets a surface's opaque region to the whole surface when its buffer
 * format has no alpha, so a window that does not cover itself is telling us it
 * is translucent and that what is behind it will show through. That is exactly
 * the set of windows worth blurring behind, and it needs no per-app
 * configuration: give an app a translucent theme and it gets frosted.
 */
static gboolean
view_wants_blur (PhocView *view)
{
  struct wlr_surface *surface = view->wlr_surface;
  pixman_box32_t full;

  if (surface == NULL || surface->current.width <= 0 || surface->current.height <= 0)
    return FALSE;

  if (!pixman_region32_not_empty (&surface->opaque_region))
    return TRUE;

  full = (pixman_box32_t) { 0, 0, surface->current.width, surface->current.height };

  return pixman_region32_contains_rectangle (&surface->opaque_region, &full) != PIXMAN_REGION_IN;
}


static void
render_layer (enum zwlr_layer_shell_v1_layer layer, PhocRenderContext *ctx)
{
  GQueue *layer_surfaces = phoc_output_get_layer_surfaces_for_layer (ctx->output, layer);

  for (GList *l = layer_surfaces->head; l; l = l->next) {
    PhocLayerSurface *layer_surface = PHOC_LAYER_SURFACE (l->data);

    if (layer_surface->mapped && phoc_layer_surface_get_blur (layer_surface) > 0) {
      /* Capture per blurred surface, not once per frame: a blurred surface has
       * to sample everything beneath it, including surfaces drawn earlier in
       * its own layer. The lock screen is exactly that case -- it sits above
       * its own background surface, and a frame-level capture would hand it
       * the apps underneath instead of the wallpaper it is covering. The stale
       * flag keeps this to one capture when nothing was drawn in between. */
      if (ctx->blur_texture == 0 || ctx->blur_stale)
        phoc_renderer_capture_blur (ctx->renderer, ctx->output, ctx);

      if (ctx->blur_texture)
        render_blur_backdrop (layer_surface, ctx);
    }

    ctx->alpha = phoc_layer_surface_get_alpha (layer_surface);
    phoc_output_layer_surface_for_each_surface (ctx->output,
                                                layer_surface,
                                                render_surface_iterator,
                                                ctx);
  }
}


static void
render_drag_icons (PhocInput *input, PhocRenderContext *ctx)
{
  ctx->alpha = 1.0;

  phoc_output_drag_icons_for_each_surface (ctx->output, input, render_surface_iterator, ctx);
}


static void
color_hsv_to_rgb (struct wlr_render_color *color)
{
  float h = color->r, s = color->g, v = color->b;

  h = fmodf (h, 360);
  if (h < 0)
    h += 360;

  int d = h / 60;
  float e = h / 60 - d;
  float a = v * (1 - s);
  float b = v * (1 - e * s);
  float c = v * (1 - (1 - e) * s);

  switch (d) {
  default:
  case 0: color->r = v, color->g = c, color->b = a; return;
  case 1: color->r = b, color->g = v, color->b = a; return;
  case 2: color->r = a, color->g = v, color->b = c; return;
  case 3: color->r = a, color->g = b, color->b = v; return;
  case 4: color->r = c, color->g = a, color->b = v; return;
  case 5: color->r = v, color->g = a, color->b = b; return;
  }
}


static struct wlr_box
phoc_box_from_touch_point (struct touch_point_data *touch_point, int width, int height)
{
  return (struct wlr_box) {
    .x = touch_point->x - width / 2.0,
    .y = touch_point->y - height / 2.0,
    .width = width,
    .height = height
  };
}

static void
render_touch_point_cb (gpointer data, gpointer user_data)
{
  struct touch_point_data *touch_point = data;
  PhocRenderContext *ctx = user_data;
  struct wlr_output *wlr_output = ctx->output->wlr_output;
  int size = TOUCH_POINT_SIZE * wlr_output->scale;
  struct wlr_render_color color = {touch_point->id * 100 + 240, 1.0, 1.0, 0.75};
  struct wlr_box point_box;

  color_hsv_to_rgb (&color);

  point_box = phoc_box_from_touch_point (touch_point, size, size);
  phoc_output_transform_box (ctx->output, &point_box);
  wlr_render_pass_add_rect (ctx->render_pass, &(struct wlr_render_rect_options){
      .box = point_box,
      .color = color,
    });

  size = TOUCH_POINT_SIZE * (1.0 - TOUCH_POINT_BORDER) * wlr_output->scale;
  point_box = phoc_box_from_touch_point (touch_point, size, size);
  phoc_output_transform_box (ctx->output, &point_box);
  wlr_render_pass_add_rect (ctx->render_pass, &(struct wlr_render_rect_options){
      .box = point_box,
      .color = COLOR_TRANSPARENT_WHITE,
    });

  point_box = phoc_box_from_touch_point (touch_point, 8 * wlr_output->scale, 2 * wlr_output->scale);
  phoc_output_transform_box (ctx->output, &point_box);
  wlr_render_pass_add_rect (ctx->render_pass, &(struct wlr_render_rect_options){
      .box = point_box,
      .color = color,
    });

  point_box = phoc_box_from_touch_point (touch_point, 2 * wlr_output->scale, 8 * wlr_output->scale);
  phoc_output_transform_box (ctx->output, &point_box);
  wlr_render_pass_add_rect (ctx->render_pass, &(struct wlr_render_rect_options){
      .box = point_box,
      .color = color,
    });
}

static void
render_touch_points (PhocRenderContext *ctx)
{
  if (G_LIKELY (ctx->output->debug_touch_points == NULL))
    return;

  g_list_foreach (ctx->output->debug_touch_points, render_touch_point_cb, ctx);
}


static void
damage_touch_point_cb (gpointer data, gpointer user_data)
{
  struct touch_point_data *touch_point = data;
  PhocOutput *output = user_data;
  struct wlr_output *wlr_output = output->wlr_output;
  int size = TOUCH_POINT_SIZE * wlr_output->scale;
  struct wlr_box box = phoc_box_from_touch_point (touch_point, size, size);
  pixman_region32_t region;

  pixman_region32_init_rect (&region, box.x, box.y, box.width, box.height);
  wlr_damage_ring_add (&output->damage_ring, &region);
  pixman_region32_fini (&region);
}

static void
damage_touch_points (PhocOutput *output)
{
  if (G_LIKELY (output->debug_touch_points == NULL))
    return;

  g_list_foreach (output->debug_touch_points, damage_touch_point_cb, output);
}

static void
view_render_to_buffer_iterator (struct wlr_surface *surface, int sx, int sy, void *_data)
{
  if (!wlr_surface_has_buffer (surface)) {
    return;
  }

  PhocServer *server = phoc_server_get_default ();
  PhocRenderer *self = phoc_server_get_renderer (server);
  struct wlr_texture *texture = wlr_surface_get_texture (surface);

  struct view_render_data *data = _data;
  PhocView *view = data->view;

  struct wlr_box geo;
  phoc_view_get_geometry (view, &geo);

  float scale = fmin (data->width / (float)geo.width,
                      data->height / (float)geo.height);

  float proj[9];
  wlr_matrix_identity (proj);
  wlr_matrix_scale (proj, scale, scale);
  wlr_matrix_translate (proj, -geo.x, -geo.y);

  struct wlr_fbox src_box;
  wlr_surface_get_buffer_source_box (surface, &src_box);

  struct wlr_box dst_box = {
    .x = sx,
    .y = sy,
    .width = surface->current.width,
    .height = surface->current.height,
  };

  float mat[9];
  wlr_matrix_project_box (mat, &dst_box, wlr_output_transform_invert (surface->current.transform), 0, proj);
  /* Droidian FIXME: handle this in wlroots? */
  wlr_matrix_project_box (mat, &dst_box, WL_OUTPUT_TRANSFORM_FLIPPED_180, 0, proj);
  wlr_render_subtexture_with_matrix (self->wlr_renderer, texture, &src_box, mat, 1.0);
}


/* FIXME: Rework when switching to wlroots 0.18.x git again */
static gboolean
phoc_renderer_render_view_to_buffer_android (PhocRenderer      *self,
                                             PhocView          *view,
                                             struct wlr_buffer *shm_buffer)
{
  struct wlr_surface *surface = view->wlr_surface;
  void *data;
  uint32_t format;
  EGLint gl_format;
  size_t stride;
  struct wlr_shm_attributes attribs;

  g_return_val_if_fail (surface, false);
  g_return_val_if_fail (self->wlr_allocator, false);
  g_return_val_if_fail (shm_buffer, false);

  if (!wlr_buffer_get_shm (shm_buffer, &attribs)) {
    format = DRM_FORMAT_ARGB8888;
  } else {
    format = attribs.format;
  }

  switch (format) {
  case DRM_FORMAT_XRGB8888:
  case DRM_FORMAT_ARGB8888:
    gl_format = GL_BGRA_EXT;
    break;
  default:
    gl_format = GL_RGBA;
    break;
  }

  int32_t width = shm_buffer->width;
  int32_t height = shm_buffer->height;

  struct wlr_egl *egl = wlr_android_renderer_get_egl (self->wlr_renderer);
  GLuint tex, fbo;

  if (!surface || !wlr_egl_make_current (egl)) {
    return false;
  }

  struct view_render_data render_data ={
    .view = view,
    .width = width,
    .height = height
  };

  glGenTextures (1, &tex);
  glBindTexture (GL_TEXTURE_2D, tex);
  glTexImage2D (GL_TEXTURE_2D, 0, gl_format, width, height, 0, gl_format, GL_UNSIGNED_BYTE, NULL);
  glBindTexture (GL_TEXTURE_2D, 0);

  glGenFramebuffers (1, &fbo);
  glBindFramebuffer (GL_FRAMEBUFFER, fbo);
  glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, tex, 0);

  wlr_renderer_begin (self->wlr_renderer, width, height);
  wlr_renderer_clear (self->wlr_renderer, (float[])COLOR_TRANSPARENT);
  wlr_surface_for_each_surface (surface, view_render_to_buffer_iterator, &render_data);
  wlr_renderer_end (self->wlr_renderer);

  if (!wlr_buffer_begin_data_ptr_access (shm_buffer,
                                         WLR_BUFFER_DATA_PTR_ACCESS_WRITE,
                                         &data, &format, &stride)) {
    return false;
  }

  wlr_renderer_read_pixels (self->wlr_renderer, format, stride, width, height, 0, 0, 0, 0, data);

  wlr_buffer_end_data_ptr_access (shm_buffer);

  glDeleteFramebuffers (1, &fbo);
  glDeleteTextures (1, &tex);
  glBindFramebuffer (GL_FRAMEBUFFER, 0);

  wlr_egl_unset_current (egl);

  return true;
}

gboolean
phoc_renderer_render_view_to_buffer (PhocRenderer      *self,
                                     PhocView          *view,
                                     struct wlr_buffer *shm_buffer)
{
  /* Do not use wlr_allocator on android */
  if (wlr_renderer_is_android(self->wlr_renderer))
    return phoc_renderer_render_view_to_buffer_android (self, view, shm_buffer);

  struct wlr_surface *surface = view->wlr_surface;
  struct wlr_buffer *buffer;
  void *data;
  uint32_t format;
  size_t stride;

  g_return_val_if_fail (surface, false);
  g_return_val_if_fail (self->wlr_allocator, false);
  g_return_val_if_fail (shm_buffer, false);

  int32_t width = shm_buffer->width;
  int32_t height = shm_buffer->height;

  struct wlr_drm_format_set fmt_set = {};
  wlr_drm_format_set_add (&fmt_set, DRM_FORMAT_ARGB8888, DRM_FORMAT_MOD_INVALID);

  const struct wlr_drm_format *fmt = wlr_drm_format_set_get (&fmt_set, DRM_FORMAT_ARGB8888);

  buffer = wlr_allocator_create_buffer (self->wlr_allocator, width, height, fmt);
  if (!buffer) {
    wlr_drm_format_set_finish (&fmt_set);
    g_return_val_if_reached (false);
  }

  struct view_render_data render_data = {
    .view = view,
    .width = width,
    .height = height
  };

  wlr_renderer_begin_with_buffer (self->wlr_renderer, buffer);
  wlr_renderer_clear (self->wlr_renderer, (float[])COLOR_TRANSPARENT);
  wlr_surface_for_each_surface (surface, view_render_to_buffer_iterator, &render_data);

  if (!wlr_buffer_begin_data_ptr_access (shm_buffer,
                                         WLR_BUFFER_DATA_PTR_ACCESS_WRITE,
                                         &data, &format, &stride)) {
    return false;
  }

  wlr_renderer_read_pixels (self->wlr_renderer,
                            DRM_FORMAT_ARGB8888, stride, width, height, 0, 0, 0, 0, data);
  wlr_renderer_end (self->wlr_renderer);

  wlr_buffer_drop (buffer);
  wlr_drm_format_set_finish (&fmt_set);

  wlr_buffer_end_data_ptr_access (shm_buffer);

  return true;
}


static void
render_damage (PhocRenderer *self, PhocRenderContext *ctx)
{
  int nrects;
  pixman_box32_t *rects;
  struct wlr_box box;

  pixman_region32_t previous_damage;

  pixman_region32_init (&previous_damage);
  pixman_region32_subtract (&previous_damage,
                            &ctx->output->damage_ring.previous[ctx->output->damage_ring.previous_idx],
                            &ctx->output->damage_ring.current);

  rects = pixman_region32_rectangles(&previous_damage, &nrects);
  for (int i = 0; i < nrects; ++i) {
    wlr_box_from_pixman_box32 (&box, rects[i]);

    phoc_output_transform_box (ctx->output, &box);
    wlr_render_pass_add_rect (ctx->render_pass, &(struct wlr_render_rect_options){
      .box = box,
      .color = COLOR_TRANSPARENT_MAGENTA,
      });
  }
  pixman_region32_fini(&previous_damage);

  rects = pixman_region32_rectangles (&ctx->output->damage_ring.current, &nrects);
  for (int i = 0; i < nrects; ++i) {
    wlr_box_from_pixman_box32 (&box, rects[i]);

    phoc_output_transform_box (ctx->output, &box);
    wlr_render_pass_add_rect (ctx->render_pass, &(struct wlr_render_rect_options){
      .box = box,
      .color = COLOR_TRANSPARENT_YELLOW,
      });
  }
}


static void
blur_state_free (PhocBlurState *state)
{
  /* At shutdown the EGL context may already be gone; deleting GL objects
   * without a current context is undefined, and the driver tears them down
   * with the context anyway. */
  if (eglGetCurrentContext () != EGL_NO_CONTEXT) {
    for (int i = 0; i < PHOC_BLUR_MAX_LEVELS; i++) {
      if (state->fbo[i])
        glDeleteFramebuffers (1, &state->fbo[i]);
      if (state->tex[i])
        glDeleteTextures (1, &state->tex[i]);
    }
    if (state->capture)
      glDeleteTextures (1, &state->capture);
  }

  g_free (state);
}


static const char *blur_vert_src =
  "attribute vec2 pos;\n"
  "attribute vec2 uv;\n"
  "varying vec2 v_uv;\n"
  "void main() {\n"
  "  v_uv = uv;\n"
  "  gl_Position = vec4(pos, 0.0, 1.0);\n"
  "}\n";

/*
 * Dual filter (Kawase) blur, the halving variant from Marius Bjørge's
 * "Bandwidth-Efficient Rendering". The width of the blur comes from how far
 * down the pyramid we go, NOT from spreading the taps: `halfpixel` stays at
 * half a texel of the target, so consecutive taps always overlap under
 * bilinear filtering and the kernel stays a smooth tent.
 *
 * Widening the taps instead is what the single resolution version of this code
 * did, and past roughly a texel of separation the five samples stop overlapping
 * and read as five discrete copies. With diagonal-only taps those copies land
 * on a diagonal lattice, which is the star pattern this replaces.
 */
static const char *blur_down_frag_src =
  "precision mediump float;\n"
  "uniform sampler2D tex;\n"
  "uniform vec2 halfpixel;\n"
  "uniform vec4 srect;\n"
  "varying vec2 v_uv;\n"
  "vec4 s(vec2 uv) { return texture2D(tex, clamp(uv, srect.xy, srect.zw)); }\n"
  "void main() {\n"
  "  vec4 c = s(v_uv) * 4.0;\n"
  "  c += s(v_uv - halfpixel);\n"
  "  c += s(v_uv + halfpixel);\n"
  "  c += s(v_uv + vec2( halfpixel.x, -halfpixel.y));\n"
  "  c += s(v_uv + vec2(-halfpixel.x,  halfpixel.y));\n"
  "  gl_FragColor = vec4(c.rgb / 8.0, 1.0);\n"
  "}\n";

/*
 * The other half of the dual filter. Unlike the down pass this one has taps on
 * the axes as well as the diagonals, weighted 1 and 2 respectively, which is
 * what makes the combined kernel isotropic. A zero halfpixel collapses it to
 * (4 + 2*4) / 12 = 1x the centre texel, so the same program draws the final
 * composite without a third shader.
 */
static const char *blur_up_frag_src =
  "precision mediump float;\n"
  "uniform sampler2D tex;\n"
  "uniform vec2 halfpixel;\n"
  "uniform vec4 srect;\n"
  "varying vec2 v_uv;\n"
  "void main() {\n"
  "  vec4 c = texture2D(tex, v_uv + vec2(-halfpixel.x * 2.0, 0.0));\n"
  "  c += texture2D(tex, v_uv + vec2(-halfpixel.x,  halfpixel.y)) * 2.0;\n"
  "  c += texture2D(tex, v_uv + vec2( 0.0, halfpixel.y * 2.0));\n"
  "  c += texture2D(tex, v_uv + vec2( halfpixel.x,  halfpixel.y)) * 2.0;\n"
  "  c += texture2D(tex, v_uv + vec2( halfpixel.x * 2.0, 0.0));\n"
  "  c += texture2D(tex, v_uv + vec2( halfpixel.x, -halfpixel.y)) * 2.0;\n"
  "  c += texture2D(tex, v_uv + vec2( 0.0, -halfpixel.y * 2.0));\n"
  "  c += texture2D(tex, v_uv + vec2(-halfpixel.x, -halfpixel.y)) * 2.0;\n"
  "  gl_FragColor = vec4(c.rgb / 12.0, 1.0);\n"
  "}\n";

/* A full target quad as a triangle strip, and the matching texture coords */
static const float blur_quad_pos[8] = { -1.0f, -1.0f,  1.0f, -1.0f,
                                        -1.0f,  1.0f,  1.0f,  1.0f };
static const float blur_quad_uv[8]  = {  0.0f,  0.0f,  1.0f,  0.0f,
                                         0.0f,  1.0f,  1.0f,  1.0f };


static GLuint
blur_compile_shader (GLenum type, const char *src)
{
  GLuint shader = glCreateShader (type);
  GLint ok = GL_FALSE;

  glShaderSource (shader, 1, &src, NULL);
  glCompileShader (shader);
  glGetShaderiv (shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[512] = { 0 };

    glGetShaderInfoLog (shader, sizeof (log) - 1, NULL, log);
    g_warning ("Failed to compile blur shader: %s", log);
    glDeleteShader (shader);
    return 0;
  }

  return shader;
}


static gboolean
blur_prog_build (PhocBlurProg *out, const char *frag_src)
{
  GLuint vert, frag;
  GLint ok = GL_FALSE;

  vert = blur_compile_shader (GL_VERTEX_SHADER, blur_vert_src);
  if (!vert)
    return FALSE;

  frag = blur_compile_shader (GL_FRAGMENT_SHADER, frag_src);
  if (!frag) {
    glDeleteShader (vert);
    return FALSE;
  }

  out->prog = glCreateProgram ();
  glAttachShader (out->prog, vert);
  glAttachShader (out->prog, frag);
  glLinkProgram (out->prog);
  glDetachShader (out->prog, vert);
  glDetachShader (out->prog, frag);
  glDeleteShader (vert);
  glDeleteShader (frag);

  glGetProgramiv (out->prog, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[512] = { 0 };

    glGetProgramInfoLog (out->prog, sizeof (log) - 1, NULL, log);
    g_warning ("Failed to link blur shader: %s", log);
    glDeleteProgram (out->prog);
    out->prog = 0;
    return FALSE;
  }

  out->attr_pos = glGetAttribLocation (out->prog, "pos");
  out->attr_uv = glGetAttribLocation (out->prog, "uv");
  out->uni_tex = glGetUniformLocation (out->prog, "tex");
  out->uni_halfpixel = glGetUniformLocation (out->prog, "halfpixel");
  out->uni_srect = glGetUniformLocation (out->prog, "srect");

  return TRUE;
}


static gboolean
blur_prog_get (PhocRenderer *self)
{
  if (self->blur_down.prog && self->blur_up.prog)
    return TRUE;

  if (self->blur_prog_failed)
    return FALSE;

  /* Only try once, a broken shader must not be recompiled every frame */
  self->blur_prog_failed = TRUE;

  if (!blur_prog_build (&self->blur_down, blur_down_frag_src))
    return FALSE;

  if (!blur_prog_build (&self->blur_up, blur_up_frag_src)) {
    glDeleteProgram (self->blur_down.prog);
    self->blur_down.prog = 0;
    return FALSE;
  }

  self->blur_prog_failed = FALSE;
  return TRUE;
}


/*
 * Whether background blur can run at all.
 *
 * A surface is only blurred when it asks to be, over the layer shell effects
 * protocol, so that request is the opt in.
 *
 * The `blur` gsetting switches the whole effect off on top of that.
 */
static gboolean
phoc_renderer_blur_enabled (PhocRenderer *self)
{
  if (!self->blur_enabled)
    return FALSE;

  /* The blur is implemented on raw GL objects, so it needs a GL renderer */
  return wlr_renderer_is_android (self->wlr_renderer) ||
         wlr_renderer_is_gles2 (self->wlr_renderer);
}

/**
 * phoc_renderer_set_blur_enabled:
 * @self: The renderer
 * @enabled: Whether to blur at all
 *
 * Set whether surfaces that ask for background blur get it. Nothing changes
 * on screen until a frame is asked for; the caller damages the outputs.
 */
void
phoc_renderer_set_blur_enabled (PhocRenderer *self, gboolean enabled)
{
  g_return_if_fail (PHOC_IS_RENDERER (self));

  self->blur_enabled = enabled;
}


static GLuint
blur_texture_new (int width, int height)
{
  GLuint tex;

  glGenTextures (1, &tex);
  glBindTexture (GL_TEXTURE_2D, tex);
  glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glBindTexture (GL_TEXTURE_2D, 0);

  return tex;
}


static PhocBlurState *
blur_state_get (PhocRenderer *self, PhocOutput *output)
{
  struct wlr_output *wlr_output = output->wlr_output;
  PhocBlurState *state = g_hash_table_lookup (self->blur_states, output);
  GLint prev_fbo = 0;

  if (state && (state->width != wlr_output->width || state->height != wlr_output->height)) {
    g_hash_table_remove (self->blur_states, output);
    state = NULL;
  }

  if (state)
    return state;

  if (self->blur_state_failed)
    return NULL;

  /* Creating the FBOs changes the framebuffer binding. The caller is inside the
   * output's render pass, which expects to keep drawing into the output, so put
   * the binding back before returning. */
  glGetIntegerv (GL_FRAMEBUFFER_BINDING, &prev_fbo);

  state = g_new0 (PhocBlurState, 1);
  state->width = wlr_output->width;
  state->height = wlr_output->height;
  /* Quarter resolution. The blur is wide enough that the lost detail does not
   * show, and it cuts the fill rate of every iteration by 16. */
  state->bw = MAX (1, state->width / 4);
  state->bh = MAX (1, state->height / 4);

  state->capture = blur_texture_new (state->width, state->height);
  for (int i = 0; i < PHOC_BLUR_MAX_LEVELS; i++) {
    state->w[i] = MAX (1, state->bw >> i);
    state->h[i] = MAX (1, state->bh >> i);
    state->tex[i] = blur_texture_new (state->w[i], state->h[i]);

    glGenFramebuffers (1, &state->fbo[i]);
    glBindFramebuffer (GL_FRAMEBUFFER, state->fbo[i]);
    glFramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                            GL_TEXTURE_2D, state->tex[i], 0);

    if (glCheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
      g_warning ("Incomplete blur framebuffer %dx%d, disabling blur", state->w[i], state->h[i]);
      blur_state_free (state);
      /* Do not retry every frame: that would reallocate a full res texture per
       * frame and repeat this warning at the frame rate. */
      self->blur_state_failed = TRUE;
      glBindFramebuffer (GL_FRAMEBUFFER, prev_fbo);
      return NULL;
    }
  }

  g_message ("Blur enabled on output '%s': %dx%d backdrop at %dx%d, %d pyramid levels",
             wlr_output->name, state->width, state->height, state->bw, state->bh,
             PHOC_BLUR_MAX_LEVELS);

  g_hash_table_insert (self->blur_states, output, state);

  glBindFramebuffer (GL_FRAMEBUFFER, prev_fbo);

  return state;
}


/* Draw a textured quad. Positions are in NDC and texture coordinates in
 * texture space, so no matrix is involved. */
static void
blur_draw_quad (const PhocBlurProg *prog,
                GLuint              texture,
                const float        *pos,
                const float        *uv,
                float               half_x,
                float               half_y,
                const float        *srect)
{
  static const float srect_full[4] = { 0.0f, 0.0f, 1.0f, 1.0f };

  if (srect == NULL)
    srect = srect_full;

  glUseProgram (prog->prog);
  glActiveTexture (GL_TEXTURE0);
  glBindTexture (GL_TEXTURE_2D, texture);
  glUniform1i (prog->uni_tex, 0);
  glUniform2f (prog->uni_halfpixel, half_x, half_y);
  glUniform4f (prog->uni_srect, srect[0], srect[1], srect[2], srect[3]);

  glVertexAttribPointer (prog->attr_pos, 2, GL_FLOAT, GL_FALSE, 0, pos);
  glVertexAttribPointer (prog->attr_uv, 2, GL_FLOAT, GL_FALSE, 0, uv);
  glEnableVertexAttribArray (prog->attr_pos);
  glEnableVertexAttribArray (prog->attr_uv);

  glDrawArrays (GL_TRIANGLE_STRIP, 0, 4);

  glDisableVertexAttribArray (prog->attr_pos);
  glDisableVertexAttribArray (prog->attr_uv);
}


/**
 * phoc_renderer_capture_blur:
 * @self: The renderer
 * @output: The output being rendered
 * @ctx:(inout): The render context
 *
 * Copy back what has been rendered into the output's framebuffer so far, blur
 * it, and store the result in @ctx for [func@render_blur_backdrop].
 *
 * Must be called from inside the output's render pass, after everything the
 * blur should pick up has been drawn and before the first blurred surface.
 * Sampling the framebuffer this way keeps us clear of the wlr_allocator, which
 * cannot back a render target on the android renderer, and it means the scene
 * is rendered once per frame rather than twice.
 */
static void
phoc_renderer_capture_blur (PhocRenderer *self, PhocOutput *output, PhocRenderContext *ctx)
{
  PhocBlurState *state;
  struct wlr_output *wlr_output = output->wlr_output;
  struct wlr_box usable;
  float srect[4];
  const float *psrect = NULL;
  guint radius;
  int levels;
  int slop, margin;
  GLint prev_fbo = 0, prev_abuf = 0, prev_vp[4] = { 0 }, prev_unit = GL_TEXTURE0;
  GLboolean prev_blend, prev_scissor;

  ctx->blur_texture = 0;
  ctx->blur_stale = FALSE;

  if (!phoc_renderer_blur_enabled (self))
    return;

  radius = phoc_output_get_blur_radius (output);
  if (radius == 0)
    return;

  if (!blur_prog_get (self))
    return;

  state = blur_state_get (self, output);
  if (!state)
    return;

  glGetIntegerv (GL_FRAMEBUFFER_BINDING, &prev_fbo);
  glGetIntegerv (GL_ARRAY_BUFFER_BINDING, &prev_abuf);
  glGetIntegerv (GL_ACTIVE_TEXTURE, &prev_unit);
  glGetIntegerv (GL_VIEWPORT, prev_vp);
  prev_blend = glIsEnabled (GL_BLEND);
  prev_scissor = glIsEnabled (GL_SCISSOR_TEST);

  glBindBuffer (GL_ARRAY_BUFFER, 0);
  glDisable (GL_BLEND);
  glDisable (GL_SCISSOR_TEST);

  /* Grab the scene rendered so far straight out of the output's framebuffer */
  glBindTexture (GL_TEXTURE_2D, state->capture);
  glCopyTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, 0, 0, state->width, state->height);
  glBindTexture (GL_TEXTURE_2D, 0);

  /*
   * How far down the pyramid to go. Reach roughly doubles per level, and level
   * 0 already works at a quarter of the output, so one of its texels is four
   * screen pixels. Radius 8 is therefore level 0 alone and each doubling of the
   * radius buys one more level.
   */
  levels = (int) roundf (log2f (MAX (radius, 8u) / 8.0f));
  levels = CLAMP (levels, 0, PHOC_BLUR_MAX_LEVELS - 1);

  /*
   * Clamp the blur's source to the output's usable area, plus a margin.
   *
   * The strips a layer surface reserves with an exclusive zone are painted
   * later in the frame than the capture is taken, and the window beneath is
   * laid out inside them and never touches them -- so under damage tracking
   * they still hold the *previous* frame's copy of the bars. Blurring that
   * drags the panel's own pixels a hundred pixels into the window, which is
   * the band along the top and the bottom.
   *
   * usable_area is the right description of this and always was. What defeated
   * it is that it is kept in logical units and cannot round-trip: at scale
   * 2.75 this output's 1080px is 392 logical units, and 392 * 2.75 is 1078.
   * Clamping to that trimmed two real columns off the right and replicated the
   * column beside them across the gap -- a band of the fix's own making. Snap
   * back to the output whenever the difference is within a scaled unit; a real
   * exclusive zone is always far larger than that.
   *
   * Deriving the region from what has been drawn does not work. The union of
   * every surface's extent goes to the whole output whenever a full screen
   * translucent surface is present, and the union of only the opaque ones
   * collapses to a sliver when the first thing drawn is the panel's own
   * background. Nor is there a bar shaped surface to subtract: phosh's panel
   * is a full screen layer surface that paints a bar and reserves a zone.
   *
   * The margin on top is because no boundary here is pixel sharp. Surfaces
   * land on fractional pixels and a scaled texture's edge is interpolated
   * against whatever is outside it; measured on this device the top bar bleeds
   * two rows past its geometry, the home bar four, and the outermost columns
   * hold a bright strip belonging to no surface at all. Three scaled units at
   * each edge costs nothing in a blur that samples across a hundred pixels.
   */
  usable = output->usable_area;
  phoc_utils_scale_box (&usable, wlr_output->scale);
  phoc_output_transform_box (output, &usable);

  slop = (int) ceilf (wlr_output->scale);
  margin = 3 * slop;

  if (usable.x <= slop) {
    usable.width += usable.x;
    usable.x = 0;
  }
  if (usable.y <= slop) {
    usable.height += usable.y;
    usable.y = 0;
  }
  if (ABS (state->width - (usable.x + usable.width)) <= slop)
    usable.width = state->width - usable.x;
  if (ABS (state->height - (usable.y + usable.height)) <= slop)
    usable.height = state->height - usable.y;

  usable.x = MAX (usable.x, 0);
  usable.y = MAX (usable.y, 0);
  usable.width  = MIN (usable.width,  state->width  - usable.x);
  usable.height = MIN (usable.height, state->height - usable.y);

  if (usable.width > 2 * margin && usable.height > 2 * margin) {
    usable.x += margin;
    usable.y += margin;
    usable.width -= 2 * margin;
    usable.height -= 2 * margin;
  }

  if (usable.width > 0 && usable.height > 0 &&
      (usable.width < state->width || usable.height < state->height)) {
    /* Buffer coords are top left based, texture coords bottom left, hence the
     * flip on y. Half a texel in on each side so the clamp lands on the last
     * fully painted pixel rather than on the boundary between the two. */
    float hx = 0.5f / state->width, hy = 0.5f / state->height;

    srect[0] = (float) usable.x / state->width + hx;
    srect[1] = 1.0f - (float) (usable.y + usable.height) / state->height + hy;
    srect[2] = (float) (usable.x + usable.width) / state->width - hx;
    srect[3] = 1.0f - (float) usable.y / state->height - hy;
    psrect = srect;
  }

  /* Downscale the full res capture into the base of the pyramid. This one step
   * is a 4x reduction, so its taps sit half a base texel out, i.e. two capture
   * texels, which covers the footprint being discarded. */
  glViewport (0, 0, state->w[0], state->h[0]);
  glBindFramebuffer (GL_FRAMEBUFFER, state->fbo[0]);
  blur_draw_quad (&self->blur_down, state->capture, blur_quad_pos, blur_quad_uv,
                  0.5f / state->w[0], 0.5f / state->h[0], psrect);

  /* Walk down, halving each time */
  for (int i = 1; i <= levels; i++) {
    glViewport (0, 0, state->w[i], state->h[i]);
    glBindFramebuffer (GL_FRAMEBUFFER, state->fbo[i]);
    blur_draw_quad (&self->blur_down, state->tex[i - 1], blur_quad_pos, blur_quad_uv,
                    0.5f / state->w[i], 0.5f / state->h[i], NULL);
  }

  /* ... and back up. Reading level i while writing level i - 1 never aliases,
   * so the levels double as their own ping-pong buffers. */
  for (int i = levels; i > 0; i--) {
    glViewport (0, 0, state->w[i - 1], state->h[i - 1]);
    glBindFramebuffer (GL_FRAMEBUFFER, state->fbo[i - 1]);
    blur_draw_quad (&self->blur_up, state->tex[i], blur_quad_pos, blur_quad_uv,
                    0.5f / state->w[i - 1], 0.5f / state->h[i - 1], NULL);
  }

  ctx->blur_texture = state->tex[0];

  /* Hand the GL state back the way the wlroots render pass left it */
  glBindFramebuffer (GL_FRAMEBUFFER, prev_fbo);
  glViewport (prev_vp[0], prev_vp[1], prev_vp[2], prev_vp[3]);
  glBindBuffer (GL_ARRAY_BUFFER, prev_abuf);
  if (prev_blend)
    glEnable (GL_BLEND);
  if (prev_scissor)
    glEnable (GL_SCISSOR_TEST);
  glUseProgram (0);
  glBindTexture (GL_TEXTURE_2D, 0);
  glActiveTexture (prev_unit);
}


/**
 * phoc_renderer_finish_frame:
 * @self: The renderer
 *
 * Release resources that had to stay alive during the frame.
 */
void
phoc_renderer_finish_frame (PhocRenderer *self)
{
  g_assert (PHOC_IS_RENDERER (self));

  g_slist_free_full (g_steal_pointer (&self->frame_textures),
                     (GDestroyNotify) wlr_texture_destroy);
  g_slist_free_full (g_steal_pointer (&self->frame_buffers),
                     (GDestroyNotify) wlr_buffer_unlock);
}


/**
 * phoc_renderer_forget_output:
 * @self: The renderer
 * @output: The output that is going away
 *
 * Drop the per output render state. The blur state owns GL textures and
 * framebuffers, and the hash table is keyed on the output pointer, so it has
 * to go before that pointer can be reused by a new output.
 */
void
phoc_renderer_forget_output (PhocRenderer *self, PhocOutput *output)
{
  g_assert (PHOC_IS_RENDERER (self));

  if (self->blur_states)
    g_hash_table_remove (self->blur_states, output);
}


/**
 * phoc_renderer_render_output:
 * @self: The renderer
 * @output: The output to render
 * @context: The render context provided by the output
 *
 * Render a given output.
 */
void
phoc_renderer_render_output (PhocRenderer *self, PhocOutput *output, PhocRenderContext *ctx)
{
  PhocServer *server = phoc_server_get_default ();
  struct wlr_output *wlr_output = output->wlr_output;
  PhocDesktop *desktop = PHOC_DESKTOP (output->desktop);
  pixman_region32_t *damage = ctx->damage;
  pixman_region32_t transformed_damage;

  g_assert (PHOC_IS_RENDERER (self));

  ctx->renderer = self;
  ctx->blur_texture = 0;
  ctx->blur_stale = FALSE;

  pixman_region32_init (&transformed_damage);

  if (!pixman_region32_not_empty (damage)) {
    // Output isn't damaged but needs buffer swap
    goto renderer_end;
  }

  pixman_region32_copy (&transformed_damage, damage);
  phoc_output_transform_damage (output, &transformed_damage);
  wlr_output_handle_damage(wlr_output, &transformed_damage);

  wlr_render_pass_add_rect (ctx->render_pass,
                            &(struct wlr_render_rect_options){
                              .box = { .width = wlr_output->width, .height = wlr_output->height },
                              .color = COLOR_BLACK,
                              .clip = &transformed_damage,
                            });

  // If a view is fullscreen on this output, render it
  if (output->fullscreen_view != NULL) {
    PhocView *view = output->fullscreen_view;

    render_view (output, view, ctx);

    // During normal rendering the xwayland window tree isn't traversed
    // because all windows are rendered. Here we only want to render
    // the fullscreen window's children so we have to traverse the tree.
#ifdef PHOC_XWAYLAND
    if (PHOC_IS_XWAYLAND_SURFACE (view)) {
      struct wlr_xwayland_surface *xsurface =
        phoc_xwayland_surface_get_wlr_surface (PHOC_XWAYLAND_SURFACE (view));
      phoc_output_xwayland_children_for_each_surface (output,
                                                      xsurface,
                                                      render_surface_iterator,
                                                      ctx);
    }
#endif

    if (phoc_output_has_shell_revealed (output)) {
      // Render top layer above fullscreen view when requested
      render_layer (ZWLR_LAYER_SHELL_V1_LAYER_TOP, ctx);
    }
  } else {
    // Render background and bottom layers under views
    render_layer (ZWLR_LAYER_SHELL_V1_LAYER_BACKGROUND, ctx);
    render_layer (ZWLR_LAYER_SHELL_V1_LAYER_BOTTOM, ctx);

    /* Render all views */
    for (GList *l = phoc_desktop_get_views (desktop)->tail; l; l = l->prev) {
      PhocView *view = PHOC_VIEW (l->data);

      if (phoc_desktop_view_is_visible (desktop, view))
        render_view (output, view, ctx);
    }
    // Render top layer above views
    // render_layer() captures the blur backdrop itself, just before each
    // surface that asked for one
    render_layer (ZWLR_LAYER_SHELL_V1_LAYER_TOP, ctx);
  }
  render_drag_icons (phoc_server_get_input (server), ctx);

  render_layer (ZWLR_LAYER_SHELL_V1_LAYER_OVERLAY, ctx);

 renderer_end:
  pixman_region32_fini (&transformed_damage);
  wlr_output_add_software_cursors_to_render_pass (wlr_output, ctx->render_pass, damage);

  render_touch_points (ctx);
  g_signal_emit (self, signals[RENDER_END], 0, ctx);
  if (G_UNLIKELY (phoc_server_check_debug_flags (server,PHOC_SERVER_DEBUG_FLAG_DAMAGE_TRACKING)))
    render_damage (self, ctx);

  damage_touch_points (output);
  g_clear_list (&output->debug_touch_points, g_free);
}


static gboolean
phoc_renderer_initable_init (GInitable    *initable,
                             GCancellable *cancellable,
                             GError      **error)
{
  PhocRenderer *self = PHOC_RENDERER (initable);

  self->wlr_renderer = wlr_renderer_autocreate (self->wlr_backend);
  if (self->wlr_renderer == NULL) {
    g_set_error (error,
                 G_FILE_ERROR, G_FILE_ERROR_FAILED,
                 "Could not create renderer");
    return FALSE;
  }

  self->wlr_allocator = wlr_allocator_autocreate (self->wlr_backend,
                                                  self->wlr_renderer);
  if (self->wlr_allocator == NULL) {
    g_set_error (error,
                 G_FILE_ERROR, G_FILE_ERROR_FAILED,
                 "Could not create allocator");
    return FALSE;
  }

  return TRUE;
}


static void
phoc_renderer_finalize (GObject *object)
{
  PhocRenderer *self = PHOC_RENDERER (object);

  phoc_renderer_finish_frame (self);
  g_clear_pointer (&self->blur_states, g_hash_table_destroy);

  if (eglGetCurrentContext () != EGL_NO_CONTEXT) {
    if (self->blur_down.prog)
      glDeleteProgram (self->blur_down.prog);
    if (self->blur_up.prog)
      glDeleteProgram (self->blur_up.prog);
  }
  self->blur_down.prog = 0;
  self->blur_up.prog = 0;

  g_clear_pointer (&self->wlr_allocator, wlr_allocator_destroy);
  g_clear_pointer (&self->wlr_renderer, wlr_renderer_destroy);

  G_OBJECT_CLASS (phoc_renderer_parent_class)->finalize (object);
}


static void
phoc_renderer_initable_iface_init (GInitableIface *iface)
{
  iface->init = phoc_renderer_initable_init;
}


static void
phoc_renderer_class_init (PhocRendererClass *klass)
{
  GObjectClass *object_class = G_OBJECT_CLASS (klass);

  object_class->get_property = phoc_renderer_get_property;
  object_class->set_property = phoc_renderer_set_property;
  object_class->finalize = phoc_renderer_finalize;

  /**
   * PhocRenderer:wlr-backend
   *
   * The wlr-backend to use for initializing the renderer
   */
  props[PROP_WLR_BACKEND] =
    g_param_spec_pointer ("wlr-backend",
                          "",
                          "",
                          G_PARAM_READWRITE | G_PARAM_CONSTRUCT_ONLY);

  g_object_class_install_properties (object_class, PROP_LAST_PROP, props);

  /**
   * PhocRenderer::render-end
   * @self: The renderer emitting the signal
   * @output: The output being rendered on
   *
   * This signal is emitted at the end of a render pass
   */
  signals[RENDER_END] = g_signal_new ("render-end",
                                      G_TYPE_FROM_CLASS (klass),
                                      G_SIGNAL_RUN_LAST,
                                      0, NULL, NULL, NULL,
                                      G_TYPE_NONE, 1,
                                      /* PhocRenderContext: */
                                      G_TYPE_POINTER);
}


static void
phoc_renderer_init (PhocRenderer *self)
{
  self->blur_enabled = TRUE;
  self->blur_states = g_hash_table_new_full (g_direct_hash, g_direct_equal,
                                             NULL, (GDestroyNotify) blur_state_free);
}


PhocRenderer *
phoc_renderer_new (struct wlr_backend *wlr_backend, GError **error)
{
  return PHOC_RENDERER (g_initable_new (PHOC_TYPE_RENDERER, NULL, error,
                                        "wlr-backend", wlr_backend,
                                        NULL));
}


struct wlr_renderer *
phoc_renderer_get_wlr_renderer (PhocRenderer *self)
{
  g_assert (PHOC_IS_RENDERER (self));

  return self->wlr_renderer;
}


struct wlr_allocator *
phoc_renderer_get_wlr_allocator (PhocRenderer *self)
{
  g_assert (PHOC_IS_RENDERER (self));

  return self->wlr_allocator;
}
