import os
import json
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_file, session, flash
import joblib
import secrets
import warnings
from datetime import datetime, timedelta
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash

warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'uploads'
DATA_FILE = 'all_data.csv'
FORECAST_FILE = 'simple_forecast_results.json'
FORECAST_CSV_FILE = 'forecast_2weeks.csv'
USERS_FILE = 'users.json'

# Authentication functions
def load_users():
    """Load users from JSON file"""
    try:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, 'r') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                users_dict = {}
                for user in data:
                    if isinstance(user, dict) and 'username' in user and 'password' in user:
                        password = user['password']
                        if not (password.startswith('pbkdf2:') or password.startswith('scrypt:') or password.startswith('bcrypt:')):
                            password = generate_password_hash(password)
                            print(f"INFO: Hashed plain text password for user: {user['username']}")
                        
                        users_dict[user['username']] = {
                            'password': password,
                            'role': user.get('role', 'user'),
                            'outlet_name': user.get('outlet_name', None),
                            'outlet_id': user.get('outlet_id', None)
                        }
                
                save_users(users_dict)
                return users_dict
            
            elif isinstance(data, dict):
                if all(isinstance(v, str) for v in data.values()):
                    users_dict = {}
                    for k, v in data.items():
                        if not (v.startswith('pbkdf2:') or v.startswith('scrypt:') or v.startswith('bcrypt:')):
                            v = generate_password_hash(v)
                        users_dict[k] = {'password': v, 'role': 'user', 'outlet_name': None, 'outlet_id': None}
                    save_users(users_dict)
                    return users_dict
                else:
                    for username, user_data in data.items():
                        if isinstance(user_data, dict) and 'password' in user_data:
                            password = user_data['password']
                            if not (password.startswith('pbkdf2:') or password.startswith('scrypt:') or password.startswith('bcrypt:')):
                                data[username]['password'] = generate_password_hash(password)
                    save_users(data)
                    return data
            else:
                return create_default_users()
        else:
            return create_default_users()
            
    except Exception as e:
        print(f"ERROR: Could not load {USERS_FILE}: {e}")
        return create_default_users()

def create_default_users():
    """Create default users file with hashed passwords"""
    default_password_hash = generate_password_hash("password123")
    default_users = {
        "admin": {
            "password": default_password_hash,
            "role": "admin",
            "outlet_name": None,
            "outlet_id": None
        }
    }
    try:
        with open(USERS_FILE, 'w') as f:
            json.dump([
                {
                    "username": "admin",
                    "password": default_password_hash,
                    "role": "admin"
                }
            ], f, indent=2)
        print(f"Created default {USERS_FILE} file with admin/password123")
    except Exception as e:
        print(f"Could not create default users file: {e}")
    return default_users

def save_users(users_dict):
    """Save users dictionary to JSON file"""
    try:
        users_list = []
        for username, user_data in users_dict.items():
            user_obj = {
                'username': username,
                'password': user_data.get('password', ''),
                'role': user_data.get('role', 'user')
            }
            if user_data.get('outlet_name'):
                user_obj['outlet_name'] = user_data['outlet_name']
            if user_data.get('outlet_id'):
                user_obj['outlet_id'] = user_data['outlet_id']
            users_list.append(user_obj)
        
        with open(USERS_FILE, 'w') as f:
            json.dump(users_list, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving users: {e}")
        return False

def verify_credentials(username, password):
    """Verify username and password"""
    users = load_users()
    user_data = users.get(username)
    if user_data and isinstance(user_data, dict):
        stored_password = user_data.get('password')
        if stored_password:
            return check_password_hash(stored_password, password)
    return False

def get_user_info(username):
    """Get full user information"""
    users = load_users()
    return users.get(username, {})

# ---------------- FLASK SETUP ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))

# ---------------- GLOBAL CACHE ----------------
forecast_cache = None

# ---------------- AUTH DECORATORS ----------------
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Access denied. Admin privileges required.', 'error')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

# ============================================================================
# V2.8 FORECASTING ENGINE - LONG TAIL OPTIMIZATION (PER OUTLET)
# ============================================================================

def process_forecast_v28(raw_df):
    """
    V2.8 Forecasting - Long Tail Optimization with Velocity-Based Segmentation
    NOW INCLUDES OUTLET_ID IN FINAL OUTPUT
    """
    global forecast_cache
    
    print("\n" + "="*100)
    print("SHOSHA PER-OUTLET FORECAST V2.8 - LONG TAIL OPTIMIZATION (WITH OUTLET_ID)")
    print("="*100)
    
    # Step 1: Prepare data
    print("\n[1/9] Loading and preparing sales data...")
    
    # Ensure proper data types
    dtype_dict = {
        'product_id': 'str',
        'product_type': 'str',
        'variant_name': 'str',
        'product_attribute': 'str',
        'quantity': 'float32',
        'outlet_name': 'str',
        'outlet_id': 'str'
    }
    
    for col, dtype in dtype_dict.items():
        if col in raw_df.columns:
            if dtype == 'str':
                raw_df[col] = raw_df[col].astype(str)
            elif dtype == 'float32':
                raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce')
    
    raw_df['created_at'] = pd.to_datetime(raw_df['created_at'], errors='coerce')
    raw_df['date'] = raw_df['created_at'].dt.date
    
    latest_date = raw_df['created_at'].max()
    earliest_date = raw_df['created_at'].min()
    total_days = (latest_date - earliest_date).days + 1
    
    print(f"   ✓ Date range: {earliest_date.date()} to {latest_date.date()}")
    print(f"   ✓ Loaded {len(raw_df):,} sales transactions")
    
    # Clean data
    raw_df = raw_df.dropna(subset=['product_id', 'variant_name', 'quantity', 'outlet_name'])
    print(f"   ✓ After cleaning: {len(raw_df):,} rows")
    
    # Step 2: Aggregate daily per outlet (including outlet_id)
    print("\n[2/9] Aggregating daily sales per outlet...")
    
    # Build grouping columns - include outlet_id if it exists
    group_cols = ['date', 'product_id', 'outlet_name']
    if 'outlet_id' in raw_df.columns:
        group_cols.append('outlet_id')
    
    daily_agg = raw_df.groupby(group_cols, as_index=False).agg({
        'quantity': 'sum'
    })
    print(f"   ✓ {len(daily_agg):,} daily records")
    
    # Step 3: Product metadata (including outlet_id)
    print("\n[3/9] Product metadata...")
    meta_cols = ['product_id']
    if 'outlet_id' in raw_df.columns:
        meta_cols.append('outlet_id')
    
    product_meta = raw_df.groupby(meta_cols, as_index=False).agg({
        'variant_name': 'first',
        'product_type': 'first',
        'product_attribute': 'first',
        'outlet_name': 'first'
    })
    product_meta['variant_name'] = product_meta['variant_name'].fillna('Unknown Product')
    product_meta['product_type'] = product_meta['product_type'].fillna('Unknown Type')
    product_meta['product_attribute'] = product_meta['product_attribute'].fillna('Unknown')
    
    # Time windows
    date_7d = (latest_date - timedelta(days=7)).date()
    date_14d = (latest_date - timedelta(days=14)).date()
    date_30d = (latest_date - timedelta(days=30)).date()
    date_90d = (latest_date - timedelta(days=90)).date()
    
    daily_agg['is_last_7d'] = daily_agg['date'] >= date_7d
    daily_agg['is_last_14d'] = daily_agg['date'] >= date_14d
    daily_agg['is_last_30d'] = daily_agg['date'] >= date_30d
    daily_agg['is_last_90d'] = daily_agg['date'] >= date_90d
    
    # Step 4: Calculate stats per outlet-product (including outlet_id)
    print("\n[4/9] Computing per-outlet forecasts...")
    
    # Build stats grouping - include outlet_id if exists
    stats_group_cols = ['product_id', 'outlet_name']
    if 'outlet_id' in daily_agg.columns:
        stats_group_cols.append('outlet_id')
    
    stats = daily_agg.groupby(stats_group_cols).agg({
        'quantity': ['sum', 'mean', 'std', 'count'],
        'is_last_7d': lambda x: daily_agg.loc[x.index, 'quantity'][daily_agg.loc[x.index, 'is_last_7d']].sum(),
        'is_last_14d': lambda x: daily_agg.loc[x.index, 'quantity'][daily_agg.loc[x.index, 'is_last_14d']].sum(),
        'is_last_30d': lambda x: daily_agg.loc[x.index, 'quantity'][daily_agg.loc[x.index, 'is_last_30d']].sum(),
        'is_last_90d': lambda x: daily_agg.loc[x.index, 'quantity'][daily_agg.loc[x.index, 'is_last_90d']].sum()
    }).reset_index()
    
    stats.columns = ['product_id', 'outlet_name'] + (['outlet_id'] if 'outlet_id' in daily_agg.columns else []) + [
        'total_qty', 'avg_daily', 'std_dev', 'sales_days', 
        'qty_7d', 'qty_14d', 'qty_30d', 'qty_90d'
    ]
    
    # Merge metadata
    merge_cols = ['product_id']
    if 'outlet_id' in product_meta.columns:
        merge_cols.append('outlet_id')
    
    stats = stats.merge(product_meta, on=merge_cols, how='left', suffixes=('', '_meta'))
    
    # Handle duplicate outlet_name column if it exists
    if 'outlet_name_meta' in stats.columns:
        stats['outlet_name'] = stats['outlet_name'].fillna(stats['outlet_name_meta'])
        stats = stats.drop(columns=['outlet_name_meta'])
    
    # Calculate averages
    stats['avg_daily_7d'] = stats['qty_7d'] / 7
    stats['avg_daily_14d'] = stats['qty_14d'] / 14
    stats['avg_daily_30d'] = stats['qty_30d'] / 30
    stats['avg_daily_90d'] = stats['qty_90d'] / 90
    stats['avg_daily_overall'] = stats['total_qty'] / total_days
    
    # V2.8 KEY: Identify velocity segments
    print("   ✓ V2.8: Segmenting products by velocity")
    stats['velocity_score'] = stats['qty_30d']
    
    # Calculate percentiles for segmentation
    p75 = stats['velocity_score'].quantile(0.75)
    p50 = stats['velocity_score'].quantile(0.50)
    p25 = stats['velocity_score'].quantile(0.25)
    
    stats['velocity_segment'] = 'Low'
    stats.loc[stats['velocity_score'] >= p25, 'velocity_segment'] = 'Medium'
    stats.loc[stats['velocity_score'] >= p50, 'velocity_segment'] = 'Medium-High'
    stats.loc[stats['velocity_score'] >= p75, 'velocity_segment'] = 'High'
    
    # V2.8: Velocity-based weighting
    print("   ✓ Applying velocity-based forecasting")
    
    # High velocity: More weight on recent trends (30d)
    high_velocity_mask = stats['velocity_segment'] == 'High'
    stats.loc[high_velocity_mask, 'weighted_avg_daily'] = (
        stats.loc[high_velocity_mask, 'avg_daily_30d'] * 0.50 +
        stats.loc[high_velocity_mask, 'avg_daily_90d'] * 0.35 +
        stats.loc[high_velocity_mask, 'avg_daily_overall'] * 0.15
    )
    
    # Medium-High velocity: Balanced
    med_high_mask = stats['velocity_segment'] == 'Medium-High'
    stats.loc[med_high_mask, 'weighted_avg_daily'] = (
        stats.loc[med_high_mask, 'avg_daily_30d'] * 0.45 +
        stats.loc[med_high_mask, 'avg_daily_90d'] * 0.40 +
        stats.loc[med_high_mask, 'avg_daily_overall'] * 0.15
    )
    
    # Medium velocity: Standard mix
    med_mask = stats['velocity_segment'] == 'Medium'
    stats.loc[med_mask, 'weighted_avg_daily'] = (
        stats.loc[med_mask, 'avg_daily_30d'] * 0.42 +
        stats.loc[med_mask, 'avg_daily_90d'] * 0.43 +
        stats.loc[med_mask, 'avg_daily_overall'] * 0.15
    )
    
    # Low velocity: More conservative (longer-term data)
    low_mask = stats['velocity_segment'] == 'Low'
    stats.loc[low_mask, 'weighted_avg_daily'] = (
        stats.loc[low_mask, 'avg_daily_30d'] * 0.35 +
        stats.loc[low_mask, 'avg_daily_90d'] * 0.50 +
        stats.loc[low_mask, 'avg_daily_overall'] * 0.15
    )
    
    # Detect momentum
    stats['momentum_7d_vs_30d'] = stats['avg_daily_7d'] / stats['avg_daily_30d'].replace(0, np.nan)
    stats['has_strong_momentum'] = (stats['momentum_7d_vs_30d'] > 2.0) & (~stats['momentum_7d_vs_30d'].isna())
    stats['has_moderate_momentum'] = (stats['momentum_7d_vs_30d'] > 1.5) & (stats['momentum_7d_vs_30d'] <= 2.0) & (~stats['momentum_7d_vs_30d'].isna())
    
    # Base forecast
    stats['forecast_14d'] = stats['weighted_avg_daily'] * 14
    
    # V2.8: Velocity-adjusted momentum boosts
    stats.loc[stats['has_strong_momentum'] & high_velocity_mask, 'forecast_14d'] *= 1.08
    stats.loc[stats['has_strong_momentum'] & ~high_velocity_mask, 'forecast_14d'] *= 1.06
    stats.loc[stats['has_moderate_momentum'] & high_velocity_mask, 'forecast_14d'] *= 1.04
    stats.loc[stats['has_moderate_momentum'] & ~high_velocity_mask, 'forecast_14d'] *= 1.02
    
    # V2.8 KEY: Maintain 1.03x bias correction
    print("   ✓ Applying 1.03x bias correction")
    stats['forecast_14d'] = stats['forecast_14d'] * 1.03
    
    # V2.8 KEY: Safety stock for high-velocity items
    stats['is_high_velocity'] = stats['velocity_segment'].isin(['High', 'Medium-High'])
    stats['is_regular_item'] = stats['qty_30d'] >= 4
    stats['has_30d_sales'] = stats['qty_30d'] >= 2
    
    # Higher minimums for high-velocity items (prevent stockouts)
    stats.loc[stats['is_high_velocity'] & (stats['qty_30d'] >= 6) & (stats['forecast_14d'] < 3), 'forecast_14d'] = 3
    stats.loc[stats['is_high_velocity'] & (stats['qty_30d'] >= 4) & (stats['forecast_14d'] < 2), 'forecast_14d'] = 2
    stats.loc[stats['is_regular_item'] & (stats['forecast_14d'] < 2), 'forecast_14d'] = 2
    stats.loc[stats['has_30d_sales'] & (stats['qty_30d'] >= 3) & (stats['forecast_14d'] < 1), 'forecast_14d'] = 1
    
    # Round
    stats['forecast_14d'] = stats['forecast_14d'].round(0).astype(int)
    
    # Step 5: Filter active products only
    print("\n[5/9] Filtering active products...")
    active_stats = stats[stats['qty_30d'] > 0].copy()
    print(f"   ✓ {len(active_stats):,} active outlet-product combinations")
    
    # Velocity distribution
    print(f"\n   Velocity Segments:")
    for segment in ['High', 'Medium-High', 'Medium', 'Low']:
        count = len(active_stats[active_stats['velocity_segment'] == segment])
        pct = count / len(active_stats) * 100 if len(active_stats) > 0 else 0
        print(f"      {segment:12s}: {count:6,d} ({pct:5.1f}%)")
    
    # Step 6: Prepare output (INCLUDING OUTLET_ID)
    print("\n[6/9] Preparing output...")
    
    # Build output columns list
    output_cols = ['outlet_name']
    if 'outlet_id' in active_stats.columns:
        output_cols.append('outlet_id')
    
    output_cols.extend([
        'product_id', 'variant_name', 'product_type', 
        'product_attribute', 'forecast_14d', 'velocity_segment'
    ])
    
    final_output = active_stats[output_cols].copy()
    
# Rename columns
    rename_dict = {
        'forecast_14d': 'recommended_stock_2w'
    }
    final_output = final_output.rename(columns=rename_dict)
    
    # Remove velocity_segment and metadata columns from final output
    columns_to_drop = ['velocity_segment']
    final_output = final_output.drop(columns=[col for col in columns_to_drop if col in final_output.columns])
    
    # Sort by outlet and recommended stock
    sort_cols = ['outlet_name']
    if 'outlet_id' in final_output.columns:
        sort_cols.append('outlet_id')
    sort_cols.append('recommended_stock_2w')
    
    final_output = final_output.sort_values(sort_cols, ascending=[True] * (len(sort_cols) - 1) + [False])
    
    # Step 7: Save files
    print("\n[7/9] Saving files...")
    final_output.to_csv(FORECAST_CSV_FILE, index=False)
    print(f"   ✓ Saved: {FORECAST_CSV_FILE} ({len(final_output):,} rows)")
    print(f"   ✓ Columns: {list(final_output.columns)}")
    
    # Step 8: Summary statistics
    print("\n[8/9] Creating summaries...")
    total_products = final_output['product_id'].nunique()
    total_outlets = final_output['outlet_name'].nunique()
    total_units = final_output['recommended_stock_2w'].sum()
    
    # Step 9: Save JSON metadata
    print("\n[9/9] Creating forecast metadata...")
    with open(FORECAST_FILE, 'w') as f:
        json.dump({
            'forecast_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_products': int(total_products),
            'total_outlets': int(total_outlets),
            'total_forecasts': len(final_output),
            'total_units': int(total_units),
            'csv_file': FORECAST_CSV_FILE,
            'version': 'v2.8-velocity-optimized-with-outlet-id'
        }, f, indent=2)
    
    forecast_cache = final_output
    
    # Print summary
    print("\n" + "="*100)
    print("V2.8 FORECAST SUMMARY - LONG TAIL OPTIMIZATION (WITH OUTLET_ID)")
    print("="*100)
    print(f"\n📊 METRICS:")
    print(f"   Total Outlets: {total_outlets}")
    print(f"   Total Products: {total_products}")
    print(f"   Total Combinations: {len(final_output):,}")
    print(f"   Total Units to Order: {total_units:,}")
    print(f"   Outlet ID Included: {'Yes' if 'outlet_id' in final_output.columns else 'No'}")
    
    print(f"\n📈 VELOCITY BREAKDOWN:")
    velocity_summary = final_output.groupby('velocity_segment').agg({
        'recommended_stock_2w': 'sum',
        'product_id': 'count'
    }).sort_values('recommended_stock_2w', ascending=False)
    print(velocity_summary)
    
    print("\n" + "="*100)
    print("✅ V2.8 COMPLETE - LONG TAIL OPTIMIZATION WITH OUTLET_ID")
    print("="*100)
    
    return final_output

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if verify_credentials(username, password):
            user_info = get_user_info(username)
            session['logged_in'] = True
            session['username'] = username
            session['role'] = user_info.get('role', 'user')
            session['outlet_name'] = user_info.get('outlet_name')
            session['outlet_id'] = user_info.get('outlet_id')
            session.permanent = True
            
            outlet_parts = []
            if user_info.get('outlet_name'):
                outlet_parts.append(user_info.get('outlet_name'))
            if user_info.get('outlet_id'):
                outlet_parts.append(f"ID: {user_info.get('outlet_id')}")
            outlet_text = f" ({', '.join(outlet_parts)})" if outlet_parts else ""
            
            flash(f'Successfully logged in as {username} [{user_info.get("role", "user")}]{outlet_text}!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password!', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    username = session.get('username', 'User')
    session.clear()
    flash(f'{username} has been logged out.', 'info')
    return redirect(url_for('login'))

@app.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    username = session.get('username')
    user_info = get_user_info(username)
    
    if request.method == 'POST':
        current_password = request.form.get('current_password', '').strip()
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if not verify_credentials(username, current_password):
            flash('Current password is incorrect!', 'error')
            return render_template('profile.html', user_info=user_info)
        
        if not new_password:
            flash('New password is required!', 'error')
            return render_template('profile.html', user_info=user_info)
        
        if len(new_password) < 6:
            flash('New password must be at least 6 characters long!', 'error')
            return render_template('profile.html', user_info=user_info)
        
        if new_password != confirm_password:
            flash('New passwords do not match!', 'error')
            return render_template('profile.html', user_info=user_info)
        
        if current_password == new_password:
            flash('New password must be different from current password!', 'warning')
            return render_template('profile.html', user_info=user_info)
        
        users = load_users()
        users[username]['password'] = generate_password_hash(new_password)
        
        if save_users(users):
            flash('Password changed successfully! Please log in with your new password.', 'success')
            session.clear()
            return redirect(url_for('login'))
        else:
            flash('Error updating password. Please try again.', 'error')
    
    return render_template('profile.html', user_info=user_info)

@app.route('/admin')
@admin_required
def admin_dashboard():
    users = load_users()
    user_list = []
    
    for username, user_data in users.items():
        user_list.append({
            'username': username,
            'role': user_data.get('role', 'user'),
            'outlet_name': user_data.get('outlet_name', ''),
            'outlet_id': user_data.get('outlet_id', ''),
            'has_outlet': bool(user_data.get('outlet_name') or user_data.get('outlet_id'))
        })
    
    user_list.sort(key=lambda x: (x['role'] != 'admin', x['username']))
    return render_template('admin_dashboard.html', users=user_list, total_users=len(user_list))

@app.route('/admin/users/create', methods=['GET', 'POST'])
@admin_required
def admin_create_user():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'user').strip()
        outlet_name = request.form.get('outlet_name', '').strip()
        outlet_id = request.form.get('outlet_id', '').strip()
        
        if not username or not password:
            flash('Username and password are required!', 'error')
            return render_template('admin_create_user.html')
        
        if len(username) < 3:
            flash('Username must be at least 3 characters long!', 'error')
            return render_template('admin_create_user.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return render_template('admin_create_user.html')
        
        if not outlet_name and not outlet_id:
            flash('Please provide at least one outlet identifier (name or ID)!', 'error')
            return render_template('admin_create_user.html')
        
        users = load_users()
        if username in users:
            flash(f'User "{username}" already exists!', 'error')
            return render_template('admin_create_user.html')
        
        user_data = {
            'password': generate_password_hash(password),
            'role': role
        }
        
        if outlet_name:
            user_data['outlet_name'] = outlet_name
        
        if outlet_id:
            user_data['outlet_id'] = outlet_id
        
        users[username] = user_data
        
        if save_users(users):
            outlet_info = []
            if outlet_name:
                outlet_info.append(f'outlet "{outlet_name}"')
            if outlet_id:
                outlet_info.append(f'ID: {outlet_id}')
            
            outlet_text = f' for {" - ".join(outlet_info)}' if outlet_info else ''
            flash(f'User "{username}" created successfully as {role}{outlet_text}!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Error saving user data!', 'error')
    
    return render_template('admin_create_user.html')

@app.route('/admin/users/<username>/edit', methods=['GET', 'POST'])
@admin_required
def admin_edit_user(username):
    users = load_users()
    
    if username not in users:
        flash(f'User "{username}" not found!', 'error')
        return redirect(url_for('admin_dashboard'))
    
    user_data = users[username]
    
    if request.method == 'POST':
        new_password = request.form.get('password', '').strip()
        new_role = request.form.get('role', 'user').strip()
        new_outlet_name = request.form.get('outlet_name', '').strip()
        new_outlet_id = request.form.get('outlet_id', '').strip()
        
        if new_password and len(new_password) < 6:
            flash('Password must be at least 6 characters long!', 'error')
            return render_template('admin_edit_user.html', user={'username': username, **user_data})
        
        if user_data.get('role') == 'admin' and new_role != 'admin':
            admin_count = sum(1 for u in users.values() if u.get('role') == 'admin')
            if admin_count <= 1:
                flash('Cannot remove the last admin user!', 'error')
                return render_template('admin_edit_user.html', user={'username': username, **user_data})
        
        if new_password:
            users[username]['password'] = generate_password_hash(new_password)
        users[username]['role'] = new_role
        users[username]['outlet_name'] = new_outlet_name if new_outlet_name else None
        users[username]['outlet_id'] = new_outlet_id if new_outlet_id else None
        
        if save_users(users):
            flash(f'User "{username}" updated successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            flash('Error saving user data!', 'error')
    
    return render_template('admin_edit_user.html', user={'username': username, **user_data})

@app.route('/admin/users/<username>/delete', methods=['POST'])
@admin_required
def admin_delete_user(username):
    users = load_users()
    
    if username not in users:
        flash(f'User "{username}" not found!', 'error')
        return redirect(url_for('admin_dashboard'))
    
    if username == session.get('username'):
        flash('You cannot delete your own account!', 'error')
        return redirect(url_for('admin_dashboard'))
    
    if users[username].get('role') == 'admin':
        admin_count = sum(1 for u in users.values() if u.get('role') == 'admin')
        if admin_count <= 1:
            flash('Cannot delete the last admin user!', 'error')
            return redirect(url_for('admin_dashboard'))
    
    del users[username]
    
    if save_users(users):
        flash(f'User "{username}" deleted successfully!', 'success')
    else:
        flash('Error saving user data!', 'error')
    
    return redirect(url_for('admin_dashboard'))

@app.route('/admin/users/<username>/reset-password', methods=['POST'])
@admin_required
def admin_reset_password(username):
    users = load_users()
    
    if username not in users:
        flash(f'User "{username}" not found!', 'error')
        return redirect(url_for('admin_dashboard'))
    
    new_password = f"temp{secrets.randbelow(10000):04d}"
    users[username]['password'] = generate_password_hash(new_password)
    
    if save_users(users):
        flash(f'Password reset for "{username}". New password: {new_password}', 'warning')
    else:
        flash('Error resetting password!', 'error')
    
    return redirect(url_for('admin_dashboard'))

# ============================================================================
# MAIN ROUTES
# ============================================================================

@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    global forecast_cache
    
    force_upload = request.args.get('upload', False)
    
    if request.method == 'GET' and os.path.exists(FORECAST_CSV_FILE) and not force_upload:
        return redirect(url_for('results', page=1))
    
    if request.method == 'POST':
        if 'file' not in request.files:
            flash("No file selected", "error")
            return render_template('upload.html')
        
        file = request.files['file']
        if file.filename == '':
            flash("No file selected", "error")
            return render_template('upload.html')
        
        if not file.filename.endswith('.csv'):
            flash("Please upload a CSV file", "error")
            return render_template('upload.html')
        
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{secrets.token_hex(8)}.csv")
        file.save(temp_path)
        
        try:
            new_df = pd.read_csv(temp_path)
            
            # Required columns for V2.8
            required_cols = ['variant_name', 'quantity', 'created_at', 'outlet_name', 'product_id']
            missing_cols = [col for col in required_cols if col not in new_df.columns]
            
            if missing_cols:
                flash(f"Missing required columns: {', '.join(missing_cols)}", "error")
                return render_template('upload.html')
            
            # Ensure product_type and product_attribute exist (can be empty)
            if 'product_type' not in new_df.columns:
                new_df['product_type'] = 'Unknown Type'
            if 'product_attribute' not in new_df.columns:
                new_df['product_attribute'] = 'Unknown'
            
            # outlet_id is optional but will be preserved if present
            
            if os.path.exists(DATA_FILE):
                old_df = pd.read_csv(DATA_FILE)
                combined_df = pd.concat([old_df, new_df]).drop_duplicates()
                flash(f"Added {len(new_df)} new records to existing data", "success")
            else:
                combined_df = new_df
                flash(f"Uploaded {len(new_df)} records", "success")
            
            combined_df.to_csv(DATA_FILE, index=False)
            
            forecast_cache = None
            
            if os.path.exists(FORECAST_CSV_FILE):
                os.remove(FORECAST_CSV_FILE)
            if os.path.exists(FORECAST_FILE):
                os.remove(FORECAST_FILE)
            
            flash("Data uploaded successfully! Generating new forecast...", "success")
            return redirect(url_for('results', page=1))
            
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
            return render_template('upload.html')
        
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return render_template('upload.html')

@app.route('/results')
@login_required
def results():
    page = int(request.args.get('page', 1))
    per_page = 50
    
    selected_outlet = request.args.get('outlet', 'all')
    filter_variant_name = request.args.get('variant_name', '')
    filter_product_id = request.args.get('product_id', '')
    filter_product_attr = request.args.get('product_attribute', '')
    filter_product_type = request.args.get('product_type', '')
    
    # Get user's outlet restriction
    user_outlet = session.get('outlet_name')
    user_outlet_id = session.get('outlet_id')
    user_role = session.get('role', 'user')
    
    if os.path.exists(FORECAST_CSV_FILE):
        try:
            forecast_df = pd.read_csv(FORECAST_CSV_FILE)
        except Exception as e:
            flash(f"Error loading forecast: {str(e)}", "error")
            return redirect(url_for('home'))
    else:
        if os.path.exists(DATA_FILE):
            try:
                raw_df = pd.read_csv(DATA_FILE)
                raw_df.columns = [c.strip() for c in raw_df.columns]
                
                print(f"\n📊 DATA LOADED FROM {DATA_FILE}:")
                print(f"   Total rows: {len(raw_df):,}")
                print(f"   Columns: {list(raw_df.columns)}")
                if 'created_at' in raw_df.columns:
                    print(f"   Date range in data: {raw_df['created_at'].min()} to {raw_df['created_at'].max()}")
                
                forecast_df = process_forecast_v28(raw_df)
                flash("Forecast V2.8 (Velocity-Optimized with Outlet ID) generated successfully!", "success")
            except Exception as e:
                flash(f"Error generating forecast: {str(e)}", "error")
                return redirect(url_for('home'))
        else:
            flash("No data available. Please upload a CSV file first.", "info")
            return redirect(url_for('home'))
    
    # SECURITY: If user has an assigned outlet, restrict to that outlet
    if user_role != 'admin':
        if user_outlet:
            forecast_df = forecast_df[forecast_df['outlet_name'].astype(str) == user_outlet]
            selected_outlet = user_outlet
        elif user_outlet_id and 'outlet_id' in forecast_df.columns:
            forecast_df = forecast_df[forecast_df['outlet_id'].astype(str) == user_outlet_id]
            selected_outlet = user_outlet_id
    
    # Get unique outlets for filter dropdown
    available_outlets = []
    if 'outlet_name' in forecast_df.columns:
        outlet_names = forecast_df['outlet_name'].dropna().unique().tolist()
        outlet_names = sorted([str(x) for x in outlet_names if str(x) != 'nan'])
        
        # If user has outlet restriction, only show their outlet
        if user_role != 'admin':
            if user_outlet:
                outlet_names = [user_outlet]
            elif user_outlet_id and 'outlet_id' in forecast_df.columns:
                outlet_ids = forecast_df['outlet_id'].dropna().unique().tolist()
                outlet_names = [str(user_outlet_id)] if str(user_outlet_id) in [str(x) for x in outlet_ids] else []
        
        available_outlets = [{'outlet_name': name} for name in outlet_names]
    
    # Apply outlet filter (only if admin or no restriction)
    filtered_df = forecast_df.copy()
    if user_role == 'admin':
        if selected_outlet != 'all' and selected_outlet:
            filtered_df = filtered_df[filtered_df['outlet_name'].astype(str) == selected_outlet]
    
    # Apply other filters
    if filter_variant_name:
        filtered_df = filtered_df[filtered_df['variant_name'].str.contains(filter_variant_name, case=False, na=False)]
    
    if filter_product_id:
        filtered_df = filtered_df[filtered_df['product_id'].str.contains(filter_product_id, case=False, na=False)]
    
    if filter_product_attr:
        filtered_df = filtered_df[filtered_df['product_attribute'].str.contains(filter_product_attr, case=False, na=False)]
    
    if filter_product_type:
        filtered_df = filtered_df[filtered_df['product_type'].str.contains(filter_product_type, case=False, na=False)]
    
    total_records = len(filtered_df)
    total_pages = max(1, (total_records - 1) // per_page + 1) if total_records else 0
    page = max(1, min(page, total_pages))
    
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_records)
    page_data = filtered_df.iloc[start_idx:end_idx]
    
    # Check if outlet_id column exists
    has_outlet_id = 'outlet_id' in page_data.columns
    
    if len(page_data) > 0:
        table_html = '<table id="forecastTable">'
        table_html += '<thead><tr>'
        table_html += '<th>Variant Name</th>'
        table_html += '<th>Outlet</th>'
        if has_outlet_id:
            table_html += '<th>Outlet ID</th>'
        table_html += '<th>Product ID</th>'
        table_html += '<th>Product Attribute</th>'
        table_html += '<th>Product Type</th>'
        table_html += '<th>Order Qty (2 Weeks)</th>'
        table_html += '</tr></thead><tbody>'
        
        for _, row in page_data.iterrows():
            variant_name = str(row.get('variant_name', '')).replace('nan', '')
            outlet_name = str(row.get('outlet_name', '')).replace('nan', '')
            outlet_id = str(row.get('outlet_id', '')).replace('nan', '') if has_outlet_id else ''
            product_id = str(row.get('product_id', '')).replace('nan', '')
            product_attribute = str(row.get('product_attribute', '')).replace('nan', '')
            product_type = str(row.get('product_type', '')).replace('nan', '')
            
            table_html += '<tr>'
            table_html += f'<td>{variant_name}</td>'
            table_html += f'<td><strong>{outlet_name}</strong></td>'
            if has_outlet_id:
                table_html += f'<td>{outlet_id}</td>'
            table_html += f'<td>{product_id}</td>'
            table_html += f'<td>{product_attribute if product_attribute != "nan" else ""}</td>'
            table_html += f'<td>{product_type}</td>'
            table_html += f'<td><strong style="color:#2563eb; font-size:1.1em;">{row.get("recommended_stock_2w", 0)}</strong></td>'
            table_html += '</tr>'
        
        table_html += '</tbody></table>'
        forecast_table = table_html
    else:
        forecast_table = "<p style='text-align:center; padding:20px; color:#666;'>No results found for the applied filters.</p>"
    
    username = session.get('username', 'Guest')
    role = session.get('role', 'user')
    user_display = f"{username} ({role})"
    user_initials = ''.join([word[0].upper() for word in username.split()[:2]]) if username != 'Guest' else 'G'
    
    return render_template(
        'results.html', 
        forecast_table=forecast_table,
        outlets=available_outlets,
        selected_outlet=selected_outlet,
        filter_variant_name=filter_variant_name,
        filter_product_id=filter_product_id,
        filter_product_attr=filter_product_attr,
        filter_product_type=filter_product_type,
        page=page, 
        total_pages=total_pages,
        total_records=total_records,
        user_display=user_display,
        user_initials=user_initials,
        forecasts=page_data.to_dict(orient='records'),
        has_outlet_id=has_outlet_id
    )

@app.route('/reset')
@login_required
def reset():
    global forecast_cache
    
    files_to_remove = [DATA_FILE, FORECAST_FILE, FORECAST_CSV_FILE]
    removed_count = 0
    
    for file_path in files_to_remove:
        if os.path.exists(file_path):
            os.remove(file_path)
            removed_count += 1
    
    forecast_cache = None
    
    flash(f"Reset complete! Removed {removed_count} files and cleared cache.", "success")
    return redirect(url_for('home'))

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/api/metrics', methods=['GET'])
@login_required
def api_metrics():
    if os.path.exists(FORECAST_FILE):
        with open(FORECAST_FILE) as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'error': 'No forecast metrics available'}), 404

@app.route('/api/results', methods=['GET'])
@login_required
def api_results():
    if os.path.exists(FORECAST_CSV_FILE):
        df = pd.read_csv(FORECAST_CSV_FILE)
        return jsonify(df.to_dict(orient='records'))
    else:
        return jsonify({'error': 'No forecast data available'}), 404

@app.route('/download/forecast')
@login_required
def download_forecast():
    if os.path.exists(FORECAST_CSV_FILE):
        return send_file(FORECAST_CSV_FILE, as_attachment=True, download_name='forecast_v2.8_velocity_optimized_with_outlet_id.csv')
    else:
        flash("No forecast file available", "error")
        return redirect(url_for('results'))

@app.route('/debug')
@login_required
def debug():
    info = {
        'data_file_exists': os.path.exists(DATA_FILE),
        'forecast_file_exists': os.path.exists(FORECAST_CSV_FILE),
        'forecast_json_exists': os.path.exists(FORECAST_FILE),
        'upload_folder_exists': os.path.exists(UPLOAD_FOLDER),
        'users_file_exists': os.path.exists(USERS_FILE),
        'forecast_version': 'v2.8-velocity-optimized-with-outlet-id'
    }
    
    if info['data_file_exists']:
        try:
            df = pd.read_csv(DATA_FILE)
            info['data_records'] = len(df)
            info['data_columns'] = list(df.columns)
        except Exception as e:
            info['data_error'] = str(e)
    
    if info['forecast_file_exists']:
        try:
            df = pd.read_csv(FORECAST_CSV_FILE)
            info['forecast_records'] = len(df)
            info['forecast_columns'] = list(df.columns)
        except Exception as e:
            info['forecast_error'] = str(e)
    
    return jsonify(info)

if __name__ == '__main__':
    try:
        users = load_users()
        print("=" * 80)
        print("FORECASTING APP V2.8 - VELOCITY OPTIMIZATION WITH OUTLET_ID")
        print("=" * 80)
        print("AVAILABLE USERS:")
        if isinstance(users, dict):
            for username, user_data in users.items():
                if isinstance(user_data, dict):
                    role = user_data.get('role', 'user')
                    outlet = user_data.get('outlet_name', 'N/A')
                    outlet_id = user_data.get('outlet_id', 'N/A')
                    print(f"  Username: {username:15} | Role: {role:10} | Outlet: {outlet:20} | ID: {outlet_id}")
        print("=" * 80)
        print("\n🚀 V2.8 Features:")
        print("  ✓ Velocity-based segmentation (High/Med-High/Med/Low)")
        print("  ✓ Dynamic weighting per velocity tier")
        print("  ✓ Velocity-adjusted momentum boosts (4-8%)")
        print("  ✓ Enhanced safety stock for high-velocity items")
        print("  ✓ 1.03x bias correction maintained")
        print("  ✓ Per-outlet forecasting")
        print("  ✓ Long tail optimization")
        print("  ✓ OUTLET_ID included in final output")
        print("=" * 80)
    except Exception as e:
        print(f"ERROR loading users: {e}")
        print("=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)