"""A hot deployment must not retain the old public controls in imported modules."""
from pathlib import Path
import runpy,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def test_stale_ui_modules_refresh_once_without_reloading_engines():
    app=runpy.run_path(str(ROOT/'app.py'),run_name='ui_reload_regression')
    import pv_presentation as presentation
    import pv_public_ui as interface
    from pv_scene_generate import generate_scene
    engine=generate_scene
    stale=lambda *a,**k:None
    presentation.PUBLIC_UI_REVISION='old';presentation.show_presentation=stale
    # Also cover the public module having a current version but stale imports.
    interface.PUBLIC_UI_REVISION=app['PUBLIC_UI_REVISION'];interface.show_presentation=stale
    loaded=app['load_public_interface']()
    assert loaded.show_presentation is presentation.show_presentation and loaded.show_presentation is not stale
    assert presentation.PUBLIC_UI_REVISION==loaded.PUBLIC_UI_REVISION==app['PUBLIC_UI_REVISION']
    interface.show_presentation=stale
    loaded=app['load_public_interface']()
    assert loaded.show_presentation is presentation.show_presentation
    unchanged=loaded.render_public_app;unchanged_view=presentation.show_presentation
    assert app['load_public_interface']().render_public_app is unchanged
    assert presentation.show_presentation is unchanged_view
    from pv_scene_generate import generate_scene as after
    assert after is engine
